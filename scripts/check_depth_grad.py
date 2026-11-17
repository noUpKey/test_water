"""Finite-difference check of the medium-aware depth gradient (dL/ds_i) in the CUDA rasterizer.

Run on a GPU machine after re-installing the package (pip install --no-use-pep517 -e .):
    python scripts/check_depth_grad.py
"""

import torch

from water_splatting.project_gaussians import project_gaussians
from water_splatting.rasterize import rasterize_gaussians


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H = W = 64
    N = 200
    fx = fy = 60.0
    cx, cy = W / 2, H / 2

    # Distinct depths with a fixed gap so a small perturbation never changes the sort order.
    z = 2.0 + 3.0 * torch.randperm(N, device=dev).float() / N  # gap = 0.015
    xy = (torch.rand(N, 2, device=dev) * 2 - 1) * z[:, None] * 0.5
    means = torch.cat([xy, z[:, None]], dim=-1)
    scales = torch.exp(torch.empty(N, 3, device=dev).uniform_(-3.0, -2.0))
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    viewmat = torch.eye(4, device=dev)[:3, :]

    xys, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
        means, scales, 1, quats, viewmat, fx, fy, cx, cy, H, W, 16, clip_thresh=0.01
    )
    xys, depths, conics = xys.detach(), depths.detach(), conics.detach()

    colors = torch.rand(N, 3, device=dev)
    opac = torch.rand(N, 1, device=dev) * 0.6 + 0.2  # stay below the 0.99 alpha clamp
    medium_rgb = torch.rand(H, W, 3, device=dev)
    medium_bs = torch.rand(H, W, 3, device=dev) * 0.5 + 0.1
    medium_attn = torch.rand(H, W, 3, device=dev) * 0.5 + 0.1
    w_obj = torch.randn(H, W, 3, device=dev)
    w_med = torch.randn(H, W, 3, device=dev)

    def loss_fn(d, scale=1.0):
        obj, _, med, _, _ = rasterize_gaussians(
            xys, torch.zeros_like(xys), d, radii, conics, num_tiles_hit, colors, opac,
            medium_rgb, medium_bs, medium_attn, H, W, 16,
            background=medium_rgb, return_alpha=True, depth_grad_scale=scale,
        )
        return (w_obj.double() * obj.double()).sum() + (w_med.double() * med.double()).sum()

    # Analytic gradient.
    d = depths.clone().requires_grad_(True)
    loss_fn(d).backward()
    g = d.grad.detach()

    # Central finite differences on the gaussians with the largest analytic gradients + random ones.
    eps = 2e-3
    idx = torch.cat([g.abs().topk(20).indices, torch.randint(0, N, (20,), device=dev)]).unique()
    fd = torch.zeros(len(idx), device=dev, dtype=torch.float64)
    with torch.no_grad():
        for k, i in enumerate(idx.tolist()):
            dp, dm = depths.clone(), depths.clone()
            dp[i] += eps
            dm[i] -= eps
            fd[k] = (loss_fn(dp) - loss_fn(dm)) / (2 * eps)

    an = g[idx].double()
    rel = (an - fd).abs() / (fd.abs().max() + 1e-12)
    print(f"{'idx':>5} {'analytic':>12} {'finite-diff':>12}")
    for i, a, f in zip(idx.tolist(), an.tolist(), fd.tolist()):
        print(f"{i:>5} {a:>12.5f} {f:>12.5f}")
    print(f"max error relative to max |grad|: {rel.max().item():.2e}")

    # scale = 0 must reproduce the original behaviour (no gradient on depths).
    d0 = depths.clone().requires_grad_(True)
    loss_fn(d0, scale=0.0).backward()
    assert d0.grad is None, "depth_grad_scale=0 should give no depth gradient"

    ok = rel.max().item() < 2e-2
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
