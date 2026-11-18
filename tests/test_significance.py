"""CPU tests for scripts/significance.py (standard library only).

    python -m pytest tests/test_significance.py      or      python tests/test_significance.py
"""

import importlib.util
import json
import os
import random
import tempfile
from pathlib import Path

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "significance.py")
_spec = importlib.util.spec_from_file_location("significance", _PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


def test_t_quantiles_match_tables():
    # Student t critical values (two-sided 95 %) from standard tables
    for df, ref in [(1, 12.706), (3, 3.182), (11, 2.201), (15, 2.131), (30, 2.042)]:
        assert abs(sig.t_ppf(0.975, df) - ref) < 2e-3, (df, sig.t_ppf(0.975, df))
    assert abs(sig.t_ppf(0.80, 15) - 0.866) < 2e-3


def test_t_p_value_roundtrip():
    for df in (3, 11, 15):
        t = sig.t_ppf(0.975, df)
        assert abs(sig.t_two_sided_p(t, df) - 0.05) < 1e-6
    assert abs(sig.t_two_sided_p(0.0, 7) - 1.0) < 1e-12


def test_signflip_exact_small_case():
    # all 4 differences positive and equal: only all-plus and all-minus reach |sum| = 4 -> p = 2/16
    assert abs(sig.signflip_p([1.0, 1.0, 1.0, 1.0]) - 2 / 16) < 1e-12
    assert sig.signflip_p([1.0, -1.0, 1.0, -1.0]) == 1.0


def test_sign_test():
    assert abs(sig.sign_test_p(8, 0) - 2 / 256) < 1e-12
    assert sig.sign_test_p(4, 4) == 1.0
    assert sig.sign_test_p(0, 0) == 1.0


def _write_runs(root, effect, noise_sd, seeds=(42, 43, 44, 45), rng_seed=0):
    rng = random.Random(rng_seed)
    scenes = {"Curasao": 29.6, "IUI3-RedSea": 31.0, "JapaneseGradens-RedSea": 23.0, "Panama": 32.3}
    for scene, level in scenes.items():
        for seed in seeds:
            shared = rng.gauss(0, noise_sd)  # same-seed runs share part of their noise
            for variant, method, shift in (("baseline", "water-splatting", 0.0), ("mdgr", "water-splatting-pseudo", effect)):
                d = Path(root) / scene / method / f"{variant}_seed{seed}"
                d.mkdir(parents=True)
                psnr = level + shared + shift + rng.gauss(0, noise_sd)
                (d / "eval.json").write_text(json.dumps({"results": {
                    "psnr": psnr, "ssim": 0.9 + 0.001 * shift, "lpips": 0.15 - 0.001 * shift}}))


def test_collect_and_detect_clear_effect():
    with tempfile.TemporaryDirectory() as tmp:
        _write_runs(tmp, effect=1.0, noise_sd=0.2)
        runs = sig.collect([tmp])
        assert len(runs) == 32
        text, data = sig.report(runs, "mdgr", "baseline")
        assert data["n_pairs"] == 16
        assert data["verdict"].startswith("SIGNIFICANT: mdgr is better"), data["verdict"]
        s = data["summary"]["psnr"]
        assert s["ci95"][0] < 1.0 < s["ci95"][1] and s["wins"] >= 14


def test_no_effect_is_not_significant():
    with tempfile.TemporaryDirectory() as tmp:
        _write_runs(tmp, effect=0.0, noise_sd=1.0, rng_seed=3)
        _, data = sig.report(sig.collect([tmp]), "mdgr", "baseline")
        assert data["verdict"].startswith("NOT SIGNIFICANT"), data["verdict"]


def test_first_root_wins_and_missing_pairs_ignored():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _write_runs(a, effect=0.5, noise_sd=0.1, seeds=(43,))
        _write_runs(b, effect=9.0, noise_sd=0.1, seeds=(42, 43))  # seed 43 duplicates are ignored
        (Path(b) / "Panama" / "water-splatting" / "baseline_seed42" / "eval.json").unlink()
        runs = sig.collect([a, b])
        ps = sig.pairs(runs, "mdgr", "baseline")
        assert len(ps) == 4 + 3  # seed 43 from root a (4 scenes) + seed 42 from root b minus Panama
        d43 = [ma["psnr"] - mb["psnr"] for _, seed, ma, mb in ps if seed == 43]
        assert all(abs(x - 0.5) < 1.0 for x in d43)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
