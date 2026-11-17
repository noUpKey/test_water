"""Region-wise error analysis of ns-eval renders: PSNR in concentric rings from the image centre to the corners.

Sparse-view artifacts concentrate where few training cameras overlap (image borders); this script quantifies
whether a method improves those regions specifically.

    python scripts/radial_metrics.py outputs/<scene>/water-splatting/<run>/renders [more render dirs ...]

Each directory must contain eval_gt_XXXX.png / eval_rgb_XXXX.png pairs (as written by `ns-eval --render-output-path`).
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def ring_psnr(render_dir: str, rings: int = 4):
    gts = sorted(glob.glob(os.path.join(render_dir, "**", "eval_gt_*.png"), recursive=True))
    if not gts:
        raise FileNotFoundError(f"no eval_gt_*.png under {render_dir}")
    sq_err = [[] for _ in range(rings)]
    for g in gts:
        gt = np.asarray(Image.open(g).convert("RGB"), dtype=np.float64) / 255.0
        pr = np.asarray(Image.open(g.replace("eval_gt_", "eval_rgb_")).convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = gt.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        ring = np.minimum((r * rings).astype(int), rings - 1)
        err = ((gt - pr) ** 2).mean(-1)
        for k in range(rings):
            sq_err[k].append(err[ring == k].mean())
    return [float(-10 * np.log10(np.mean(e))) for e in sq_err], len(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--rings", type=int, default=4)
    args = ap.parse_args()
    print(f"{'render dir':60s} {'n':>3s}  " + "  ".join(f"ring{k}" for k in range(args.rings)) + "   (centre -> corner, PSNR dB)")
    for d in args.dirs:
        psnr, n = ring_psnr(d, args.rings)
        print(f"{d[-60:]:60s} {n:3d}  " + "  ".join(f"{p:5.2f}" for p in psnr))


if __name__ == "__main__":
    main()
