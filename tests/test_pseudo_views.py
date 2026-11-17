"""CPU unit tests for water_splatting/pseudo_views.py (no CUDA / nerfstudio needed).

    python -m pytest tests/test_pseudo_views.py      or      python tests/test_pseudo_views.py
"""

import importlib.util
import math
import os

import torch

_PATH = os.path.join(os.path.dirname(__file__), "..", "water_splatting", "pseudo_views.py")
_spec = importlib.util.spec_from_file_location("pseudo_views", _PATH)
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)


def _random_rotations(n, g):
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    return pv.quat_to_rotmat(q)


def _trajectory(n=12, g=None):
    """A forward-facing arc of cameras, similar to an underwater survey."""
    t = torch.linspace(-0.5, 0.5, n)
    pos = torch.stack([t * 2, 0.1 * torch.sin(3 * t), torch.zeros_like(t)], -1)
    yaw = 0.3 * t
    R = pv.axis_angle_to_rotmat(torch.tensor([[0.0, 1.0, 0.0]]).repeat(n, 1), yaw)
    return torch.cat([R, pos[:, :, None]], -1)


def test_quat_roundtrip():
    g = torch.Generator().manual_seed(0)
    R = _random_rotations(500, g)
    R2 = pv.quat_to_rotmat(pv.rotmat_to_quat(R))
    assert torch.allclose(R, R2, atol=1e-5)


def test_slerp_endpoints_and_midpoint():
    g = torch.Generator().manual_seed(1)
    R0, R1 = _random_rotations(50, g), _random_rotations(50, g)
    q0, q1 = pv.rotmat_to_quat(R0), pv.rotmat_to_quat(R1)
    a = pv.quat_to_rotmat(pv.quat_slerp(q0, q1, torch.zeros(50)))
    b = pv.quat_to_rotmat(pv.quat_slerp(q0, q1, torch.ones(50)))
    assert torch.allclose(a, R0, atol=1e-5) and torch.allclose(b, R1, atol=1e-5)
    # midpoint is equidistant (geodesic angle) from both ends
    m = pv.quat_to_rotmat(pv.quat_slerp(q0, q1, torch.full((50,), 0.5)))

    def ang(A, B):
        tr = (A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.acos(((tr - 1) / 2).clamp(-1, 1))

    assert torch.allclose(ang(R0, m), ang(m, R1), atol=1e-4)


def test_axis_angle():
    R = pv.axis_angle_to_rotmat(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2]))
    assert torch.allclose(R[0] @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_pose_sampler_valid_and_local():
    c2w = _trajectory(12)
    g = torch.Generator().manual_seed(3)
    poses, src = pv.sample_pseudo_poses(c2w, 40, g, rot_jitter_deg=5.0, trans_jitter=0.5, backward_jitter=0.5)
    assert poses.shape == (40, 3, 4) and src.shape == (40,)
    R = poses[:, :3, :3]
    eye = torch.eye(3).expand(40, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5), "rotations must be orthonormal"
    assert torch.allclose(torch.det(R), torch.ones(40), atol=1e-5), "rotations must be proper"
    # every pseudo pose stays near the trajectory: within ~2 camera spacings and 15 degrees of its source camera
    spacing = (c2w[1:, :3, 3] - c2w[:-1, :3, 3]).norm(dim=-1).median()
    d = (poses[:, :3, 3] - c2w[src, :3, 3]).norm(dim=-1)
    assert (d <= 2.0 * spacing + 1e-6).all()
    cosang = (-poses[:, :3, 2] * -c2w[src, :3, 2]).sum(-1)
    assert (cosang >= math.cos(math.radians(15))).all()
    # deterministic for a fixed seed
    p2, s2 = pv.sample_pseudo_poses(c2w, 40, torch.Generator().manual_seed(3))
    assert torch.equal(poses, p2) and torch.equal(src, s2)


def test_nearest_reference_identity():
    c2w = _trajectory(10)
    assert torch.equal(pv.nearest_reference(c2w, c2w), torch.arange(10))


def _fake_render(h=24, w=32, seed=0):
    """Synthesises a physically consistent decomposition: J (clear), A in (0,1], B (medium)."""
    g = torch.Generator().manual_seed(seed)
    J = torch.rand(h, w, 3, generator=g) * 2.5  # restored radiance can exceed 1
    A = 0.05 + 0.9 * torch.rand(h, w, 3, generator=g)
    B = 0.3 * torch.rand(h, w, 3, generator=g)
    J[:3, :3] = 0.0  # a few empty pixels (no gaussians)
    obj = A * J
    return obj + B, obj, J, B


def test_identity_fixer_reproduces_render_clear_space():
    """Core invariant: with an identity fixer the medium-decoupled target equals the current render."""
    rgb, obj, J, B = _fake_render()
    rgb = rgb.clamp(0, 1)
    inp, _, s = pv.prepare_fixer_input(rgb, obj, J, B, fix_space="clear")
    assert inp.min() >= 0 and inp.max() <= 1
    fixed = pv.IdentityFixer()([inp])[0]
    tgt = pv.compose_target(fixed, rgb, obj, J, B, s, fix_space="clear", do_color_match=False)
    assert torch.allclose(tgt, rgb, atol=1e-5), (tgt - rgb).abs().max()


def test_identity_fixer_with_color_match_is_stable():
    rgb, obj, J, B = _fake_render(seed=1)
    rgb = rgb.clamp(0, 1)
    inp, _, s = pv.prepare_fixer_input(rgb, obj, J, B, fix_space="clear")
    tgt = pv.compose_target(pv.IdentityFixer()([inp])[0], rgb, obj, J, B, s, fix_space="clear", do_color_match=True)
    assert (tgt - rgb).abs().mean() < 1e-3


def test_identity_fixer_underwater_space():
    rgb, obj, J, B = _fake_render(seed=2)
    rgb = rgb.clamp(0, 1)
    inp, _, s = pv.prepare_fixer_input(rgb, obj, J, B, fix_space="underwater")
    tgt = pv.compose_target(pv.IdentityFixer()([inp])[0], rgb, obj, J, B, s, fix_space="underwater", do_color_match=False)
    assert torch.allclose(tgt, rgb, atol=1e-6)


def test_fix_only_changes_object_component():
    """A change made by the fixer in clear space is attenuated by A and never alters the medium term B."""
    rgb, obj, J, B = _fake_render(seed=3)
    inp, _, s = pv.prepare_fixer_input(rgb.clamp(0, 1), obj, J, B, fix_space="clear")
    fixed = (inp + 0.1).clamp(0, 1)
    tgt = pv.compose_target(fixed, rgb, obj, J, B, s, fix_space="clear", do_color_match=False)
    A = pv.effective_attenuation(obj, J)
    expected = (A * torch.where(inp >= 0.999, J, fixed * s) + B).clamp(0, 1)
    assert torch.allclose(tgt, expected, atol=1e-5)
    empty = J.sum(-1) == 0
    assert torch.allclose(tgt[empty], B[empty].clamp(0, 1)), "pixels without gaussians must keep the medium only"


def test_restore_with_medium_inverts_formation_model():
    rgb, obj, J, B = _fake_render(seed=4)
    J_ref = pv.restore_with_medium(rgb, obj, J, B)
    A = pv.effective_attenuation(obj, J)
    ok = A > 0.05
    assert torch.allclose(J_ref[ok], J[ok], atol=1e-4)


def test_effective_attenuation_bounds():
    rgb, obj, J, B = _fake_render(seed=5)
    A = pv.effective_attenuation(obj, J)
    assert A.min() >= 0 and A.max() <= 1


class _FakePipe:
    """Stands in for DifixPipeline: raises CUDA OOM above `limit` pixels, otherwise returns the input image + 0.1."""

    def __init__(self, limit):
        self.limit, self.sizes = limit, []

    def to(self, device):
        return self

    def __call__(self, prompt, image, height, width, **kw):
        import numpy as np
        from PIL import Image
        self.sizes.append((height, width))
        if height * width > self.limit:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        arr = np.clip(np.asarray(image, dtype=np.float32) + 25.5, 0, 255).astype(np.uint8)
        return type("Out", (), {"images": [Image.fromarray(arr)]})()


def test_difix_oom_retries_smaller_then_falls_back():
    img = torch.full((600, 900, 3), 0.5)
    # fits after one downscale (768 -> 576 long side)
    f = pv.DifixFixer(device="cpu", max_side=768); f.pipe = _FakePipe(limit=576 * 384)
    out = f([img], [None])[0]
    assert f.pipe.sizes[0][1] == 768 and f.pipe.sizes[-1][1] == 576, f.pipe.sizes
    assert out.shape == img.shape and torch.allclose(out, torch.full_like(img, 0.6), atol=0.01)
    # never fits -> returned unchanged, no exception
    f = pv.DifixFixer(device="cpu", max_side=768); f.pipe = _FakePipe(limit=0)
    out = f([img], [None])[0]
    assert torch.equal(out, img) and min(w for _, w in f.pipe.sizes) >= 384


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
