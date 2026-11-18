"""Paired significance test of one variant against another over repeated training runs.

The unit of replication is a training run. Variant A and variant B trained on the same scene with the same seed form
one pair, and the test asks whether the mean paired difference A - B is distinguishable from run-to-run noise.

    python scripts/significance.py --roots outputs/ prior_session/outputs --a mdgr --b baseline [--out results/sig]

Runs are found as <root>/.../<scene>/<method>/<variant>_seed<k>/eval.json (the output of `ns-eval`). When the same
(scene, variant, seed) appears under several roots, the first root wins.

Decision rule (fixed before looking at the multi-seed data):
  primary endpoint = mean paired PSNR difference over all (scene, seed) pairs;
  A is significantly better than B if the exact sign-flip permutation p < 0.05 (two-sided)
  and the 95 % t confidence interval of the mean difference excludes 0. SSIM and LPIPS are secondary.

Pure standard library (no numpy / scipy), so it runs in any Python >= 3.8.
"""

import argparse
import itertools
import json
import math
import random
import re
from pathlib import Path
from statistics import mean, stdev

METRICS = {"psnr": +1, "ssim": +1, "lpips": -1}  # +1: higher is better, -1: lower is better
PRIMARY = "psnr"


# ----------------------------------------------------------------------------------------------- data


def collect(roots):
    """{(scene, variant, seed): {metric: value}} from every eval.json under the given roots."""
    runs = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for ev in sorted(root.rglob("eval.json")):
            m = re.fullmatch(r"(.+)_seed(\d+)", ev.parent.name)
            if not m:
                continue
            key = (ev.parents[2].name, m.group(1), int(m.group(2)))
            if key in runs:
                continue
            res = json.loads(ev.read_text())["results"]
            runs[key] = {k: float(res[k]) for k in METRICS if k in res}
    return runs


def pairs(runs, a, b):
    """[(scene, seed, metrics_a, metrics_b)] for every (scene, seed) that has both variants."""
    out = []
    for (scene, variant, seed), ma in sorted(runs.items()):
        if variant == a and (scene, b, seed) in runs:
            out.append((scene, seed, ma, runs[(scene, b, seed)]))
    return out


# ----------------------------------------------------------------------------------------------- statistics


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-15:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t statistic."""
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_ppf(q, df):
    """Quantile of Student's t distribution for q in (0.5, 1)."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - 0.5 * t_two_sided_p(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def signflip_p(d, exact_max_n=18, n_mc=200_000, seed=0):
    """Two-sided permutation p-value for mean(d) = 0, flipping the sign of each paired difference."""
    n, obs = len(d), abs(sum(d)) - 1e-12
    if n <= exact_max_n:
        hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs for signs in itertools.product((1, -1), repeat=n))
        return hits / 2 ** n
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs for _ in range(n_mc))
    return (hits + 1) / (n_mc + 1)


def sign_test_p(wins, losses):
    """Two-sided exact binomial (sign) test; ties are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def summarize(d, better_sign):
    """Statistics of paired differences d = A - B. better_sign = +1 if larger is better for this metric."""
    n = len(d)
    out = {"n": n, "mean": mean(d) if n else float("nan")}
    wins = sum(1 for x in d if x * better_sign > 0)
    losses = sum(1 for x in d if x * better_sign < 0)
    out.update(wins=wins, losses=losses, sign_p=sign_test_p(wins, losses))
    if n >= 2:
        sd = stdev(d)
        se = sd / math.sqrt(n)
        tq = t_ppf(0.975, n - 1)
        out.update(sd=sd, se=se, ci95=(out["mean"] - tq * se, out["mean"] + tq * se),
                   t_p=t_two_sided_p(out["mean"] / se, n - 1) if se > 0 else (1.0 if out["mean"] == 0 else 0.0),
                   perm_p=signflip_p(d),
                   # smallest true effect this design detects with 80 % power at alpha = 0.05 (two-sided)
                   mde80=(tq + t_ppf(0.80, n - 1)) * se)
    return out


def pooled_seed_sd(runs, variant, metric):
    """Run-to-run SD of one variant within a scene, pooled over scenes with >= 2 seeds."""
    by_scene = {}
    for (scene, v, _), m in runs.items():
        if v == variant and metric in m:
            by_scene.setdefault(scene, []).append(m[metric])
    num = sum((len(x) - 1) * stdev(x) ** 2 for x in by_scene.values() if len(x) >= 2)
    den = sum(len(x) - 1 for x in by_scene.values() if len(x) >= 2)
    return math.sqrt(num / den) if den else float("nan")


# ----------------------------------------------------------------------------------------------- report


def report(runs, a, b):
    ps = pairs(runs, a, b)
    lines = [f"# Paired significance: {a} vs {b}", ""]
    if not ps:
        return "\n".join(lines + ["No (scene, seed) pairs with both variants."]), {}
    scenes = sorted({p[0] for p in ps})
    seeds = sorted({p[1] for p in ps})
    lines += [f"{len(ps)} pairs: {len(scenes)} scenes x seeds {seeds}. Difference = {a} - {b}.", ""]

    lines += ["## Per pair (PSNR, dB)", "", f"| scene | seed | {a} | {b} | diff |", "|---|---|---|---|---|"]
    for scene, seed, ma, mb in ps:
        lines.append(f"| {scene} | {seed} | {ma['psnr']:.3f} | {mb['psnr']:.3f} | {ma['psnr'] - mb['psnr']:+.3f} |")

    lines += ["", "## Per scene (mean diff over seeds)", "", "| scene | n | PSNR diff | SSIM diff | LPIPS diff |",
              "|---|---|---|---|---|"]
    for s in scenes:
        sub = [p for p in ps if p[0] == s]
        cells = []
        for k in METRICS:
            d = [ma[k] - mb[k] for _, _, ma, mb in sub]
            cells.append(f"{mean(d):+.4f}" + (f" ± {stdev(d):.4f}" if len(d) > 1 else ""))
        lines.append(f"| {s} | {len(sub)} | " + " | ".join(cells) + " |")

    summary = {}
    lines += ["", "## Overall", "",
              "| metric | mean diff | 95% CI | wins/losses | perm. p | t-test p | sign p | detectable (80% power) |",
              "|---|---|---|---|---|---|---|---|"]
    for k, sgn in METRICS.items():
        d = [ma[k] - mb[k] for _, _, ma, mb in ps if k in ma and k in mb]
        s = summarize(d, sgn)
        summary[k] = s
        if s["n"] >= 2:
            lines.append(f"| {k}{' (primary)' if k == PRIMARY else ''} | {s['mean']:+.4f} | "
                         f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['wins']}/{s['losses']} | "
                         f"{s['perm_p']:.4f} | {s['t_p']:.4f} | {s['sign_p']:.4f} | ±{s['mde80']:.4f} |")
        else:
            lines.append(f"| {k} | {s['mean']:+.4f} | n < 2 | {s['wins']}/{s['losses']} | | | | |")

    noise = {v: pooled_seed_sd(runs, v, PRIMARY) for v in (a, b)}
    lines += ["", "Run-to-run PSNR SD within a scene (pooled over scenes): "
              + ", ".join(f"{v} {sd:.3f} dB" if sd == sd else f"{v} n/a (one seed)" for v, sd in noise.items()), ""]

    p = summary[PRIMARY]
    if p["n"] < 2:
        verdict = "Not enough pairs for a test."
    else:
        sig = p["perm_p"] < 0.05 and (p["ci95"][0] > 0 or p["ci95"][1] < 0)
        better = p["mean"] * METRICS[PRIMARY] > 0
        if sig:
            verdict = (f"SIGNIFICANT: {a} is {'better' if better else 'worse'} than {b} on PSNR by "
                       f"{p['mean']:+.3f} dB (95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}], perm. p = {p['perm_p']:.4f}).")
        else:
            verdict = (f"NOT SIGNIFICANT: mean PSNR difference {p['mean']:+.3f} dB, 95% CI [{p['ci95'][0]:+.3f}, "
                       f"{p['ci95'][1]:+.3f}], perm. p = {p['perm_p']:.4f}. With {p['n']} pairs this design only "
                       f"reliably detects effects of about ±{p['mde80']:.2f} dB or larger.")
    lines += ["## Verdict (primary endpoint: PSNR)", "", verdict]
    return "\n".join(lines), {"a": a, "b": b, "n_pairs": len(ps), "scenes": scenes, "seeds": seeds,
                              "summary": summary, "seed_sd_psnr": noise, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="directories searched recursively for eval.json")
    ap.add_argument("--a", default="mdgr", help="variant under test")
    ap.add_argument("--b", default="baseline", help="reference variant")
    ap.add_argument("--out", default="", help="write <out>.md and <out>.json")
    args = ap.parse_args()

    runs = collect(args.roots)
    text, data = report(runs, args.a, args.b)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(text + "\n")
        out.with_suffix(".json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
