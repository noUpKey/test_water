"""Medium-decoupled generative refinement (MDGR) for WaterSplatting.

Idea
----
With only 15-25 training views, regions seen by few cameras (image borders, near-field seabed) are reconstructed
poorly. Diffusion "fixers" such as Difix (Wu et al., CVPR 2025) can repair such artifacts in renders of novel
viewpoints, and the repaired renders can be distilled back into the 3D representation. However these models are
trained on *in-air* imagery: applied to an underwater render they see an out-of-distribution, hazy, colour-cast
image and tend to "fix" the water itself (removing backscatter, shifting colours), which then fights the physical
medium model.

WaterSplatting already decomposes every render into

    I = J_att + B,   J_att = sum_i T_i a_i c_i exp(-sigma_attn s_i),   J = sum_i T_i a_i c_i   (clear / restored)

so we run the fixer on the *restored* (medium-free) image J, which is close to the fixer's training distribution,
and re-apply the model's own medium afterwards:

    A      = J_att / J                    (per-pixel, per-channel effective attenuation, in [0, 1])
    I_tgt  = A * Fix(J) + B

The resulting pseudo-target is consistent with the learned water by construction: if the fixer returns its input
unchanged, I_tgt equals the current render exactly (see tests/test_pseudo_views.py). Pseudo-views only use poses
derived from *training* cameras (interpolation + perturbation), never test poses or test images.

This module is pure torch (no nerfstudio / CUDA imports) so it can be unit-tested on CPU.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------------------------------------------------
# Rotation helpers (quaternions are (w, x, y, z))
# --------------------------------------------------------------------------------------------------------------------


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Converts rotation matrices (..., 3, 3) to unit quaternions (..., 4) in (w, x, y, z) order."""
    m = R.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    # Four candidate (squared, scaled) components; pick the numerically largest one per matrix (Shepperd's method).
    cand = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    best = cand.argmax(dim=-1)
    q = torch.empty(m.shape[0], 4, dtype=m.dtype, device=m.device)
    for k in range(4):
        sel = best == k
        if not sel.any():
            continue
        s = torch.sqrt(cand[sel, k].clamp_min(1e-12)) * 2.0  # s = 4 * |component|
        if k == 0:
            q[sel] = torch.stack([0.25 * s, (m21 - m12)[sel] / s, (m02 - m20)[sel] / s, (m10 - m01)[sel] / s], -1)
        elif k == 1:
            q[sel] = torch.stack([(m21 - m12)[sel] / s, 0.25 * s, (m01 + m10)[sel] / s, (m02 + m20)[sel] / s], -1)
        elif k == 2:
            q[sel] = torch.stack([(m02 - m20)[sel] / s, (m01 + m10)[sel] / s, 0.25 * s, (m12 + m21)[sel] / s], -1)
        else:
            q[sel] = torch.stack([(m10 - m01)[sel] / s, (m02 + m20)[sel] / s, (m12 + m21)[sel] / s, 0.25 * s], -1)
    q = F.normalize(q, dim=-1)
    q = torch.where(q[:, :1] < 0, -q, q)  # canonical hemisphere
    return q.reshape(*R.shape[:-2], 4)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Converts quaternions (..., 4) in (w, x, y, z) order to rotation matrices (..., 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between unit quaternions q0, q1 (N, 4) at fractions t (N,)."""
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # shortest path
    dot = dot.abs().clamp(max=1.0)
    t = t.reshape(-1, 1).to(q0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    use_lerp = sin_theta < 1e-6
    w0 = torch.where(use_lerp, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-12))
    w1 = torch.where(use_lerp, t, torch.sin(t * theta) / sin_theta.clamp_min(1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def axis_angle_to_rotmat(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues' formula. axis (N, 3) (normalised internally), angle (N,) in radians -> (N, 3, 3)."""
    axis = F.normalize(axis, dim=-1)
    half = (0.5 * angle).reshape(-1, 1)
    q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    return quat_to_rotmat(q)


# --------------------------------------------------------------------------------------------------------------------
# Pseudo-view pose sampling (nerfstudio / OpenGL camera convention: x right, y up, camera looks along -z)
# --------------------------------------------------------------------------------------------------------------------


def _median_nn_distance(positions: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns (median nearest-neighbour distance, pairwise distance matrix) of camera centres (N, 3)."""
    dist = torch.cdist(positions, positions)
    n = positions.shape[0]
    if n < 2:
        return positions.new_tensor(1.0), dist
    masked = dist + torch.eye(n, dtype=dist.dtype, device=dist.device) * 1e10
    return masked.min(dim=1).values.median().clamp_min(1e-6), dist


def sample_pseudo_poses(
    c2w: Tensor,
    num: int,
    generator: Optional[torch.Generator] = None,
    interp_ratio: float = 0.5,
    k_neighbors: int = 3,
    interp_range: Tuple[float, float] = (0.25, 0.75),
    rot_jitter_deg: float = 5.0,
    trans_jitter: float = 0.5,
    backward_jitter: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Samples pseudo camera poses around the training trajectory.

    Two kinds of poses are generated:
      * interpolated: between a training camera and one of its ``k_neighbors`` nearest training cameras
        (position lerp + rotation slerp), filling the gaps between training views;
      * perturbed: a training camera rotated by up to ``rot_jitter_deg`` (yaw / pitch) and translated by up to
        ``trans_jitter`` x (median camera spacing) in its image plane, plus a *backward* step of up to
        ``backward_jitter`` x spacing. Stepping back widens the frustum so that the poorly observed image borders of
        the training views appear inside the pseudo views -- exactly the regions with the largest test error.

    Args:
        c2w: training camera-to-world matrices (N, 3, 4) or (N, 4, 4).
        num: number of pseudo poses to generate.
    Returns:
        poses (num, 3, 4) and the index (num,) of the training camera each pose was derived from
        (used to copy intrinsics).
    """
    c2w = c2w[:, :3, :4].float()
    n = c2w.shape[0]
    if num <= 0:
        return c2w.new_zeros(0, 3, 4), torch.zeros(0, dtype=torch.long)
    R = c2w[:, :3, :3]
    p = c2w[:, :3, 3]
    spacing, dist = _median_nn_distance(p)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    num_interp = int(round(num * interp_ratio)) if n >= 2 else 0
    num_pert = num - num_interp
    poses, srcs = [], []

    if num_interp > 0:
        i = torch.randint(0, n, (num_interp,), generator=generator)
        k = min(k_neighbors, n - 1)
        nn_idx = (dist + torch.eye(n) * 1e10).topk(k, dim=1, largest=False).indices  # (n, k)
        j = nn_idx[i, torch.randint(0, k, (num_interp,), generator=generator)]
        t = interp_range[0] + (interp_range[1] - interp_range[0]) * rand(num_interp)
        q = quat_slerp(rotmat_to_quat(R[i]), rotmat_to_quat(R[j]), t)
        pos = (1 - t[:, None]) * p[i] + t[:, None] * p[j]
        poses.append(torch.cat([quat_to_rotmat(q), pos[:, :, None]], dim=-1))
        srcs.append(torch.where(t < 0.5, i, j))

    if num_pert > 0:
        i = torch.randint(0, n, (num_pert,), generator=generator)
        Ri = R[i]
        # rotation about a random axis in the camera's image plane (yaw / pitch, no roll)
        phi = 2 * math.pi * rand(num_pert)
        axis_cam = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=-1)
        axis_world = (Ri @ axis_cam[:, :, None]).squeeze(-1)
        angle = math.radians(rot_jitter_deg) * rand(num_pert)
        R_new = axis_angle_to_rotmat(axis_world, angle) @ Ri
        # translation: in-plane jitter + backward step (+z is "backwards" in the OpenGL camera frame)
        off_cam = torch.stack(
            [
                trans_jitter * (2 * rand(num_pert) - 1),
                trans_jitter * (2 * rand(num_pert) - 1),
                backward_jitter * rand(num_pert),
            ],
            dim=-1,
        ) * spacing
        pos = p[i] + (Ri @ off_cam[:, :, None]).squeeze(-1)
        poses.append(torch.cat([R_new, pos[:, :, None]], dim=-1))
        srcs.append(i)

    perm = torch.randperm(num, generator=generator)
    return torch.cat(poses, 0)[perm], torch.cat(srcs, 0)[perm]


def nearest_reference(c2w_train: Tensor, c2w_query: Tensor, angle_weight: float = 2.0) -> Tensor:
    """For each query pose returns the index of the most similar training camera (position + viewing direction)."""
    pt, pq = c2w_train[:, :3, 3].float(), c2w_query[:, :3, 3].float()
    spacing, _ = _median_nn_distance(pt)
    ft = -c2w_train[:, :3, 2].float()  # forward = -z
    fq = -c2w_query[:, :3, 2].float()
    pos_term = torch.cdist(pq, pt) / spacing
    ang_term = 1.0 - F.normalize(fq, dim=-1) @ F.normalize(ft, dim=-1).T
    return (pos_term + angle_weight * ang_term / (1.0 - math.cos(math.radians(10.0)))).argmin(dim=1)


# --------------------------------------------------------------------------------------------------------------------
# Medium-decoupled target construction
# --------------------------------------------------------------------------------------------------------------------


def exposure_scale(img: Tensor, q: float = 0.99) -> Tensor:
    """Robust exposure normaliser for the (unbounded) restored image: the q-quantile of its values, >= 1e-3."""
    flat = img.reshape(-1).float()
    if flat.numel() > 2_000_000:  # torch.quantile has an input-size limit
        flat = flat[torch.randperm(flat.numel(), device=flat.device)[:2_000_000]]
    return torch.quantile(flat, q).clamp_min(1e-3)


def effective_attenuation(rgb_object: Tensor, rgb_clear: Tensor, eps: float = 1e-4) -> Tensor:
    """A = J_att / J, the per-pixel transmittance-weighted mean of exp(-sigma_attn s_i); 0 where J is ~0."""
    A = rgb_object / rgb_clear.clamp_min(eps)
    return torch.where(rgb_clear > eps, A, torch.zeros_like(A)).clamp(0.0, 1.0)


def color_match(src: Tensor, ref: Tensor, eps: float = 1e-6) -> Tensor:
    """Matches per-channel mean / std of ``src`` (H, W, 3) to ``ref``: removes global colour drift of the fixer."""
    s_mu, s_std = src.mean(dim=(0, 1)), src.std(dim=(0, 1))
    r_mu, r_std = ref.mean(dim=(0, 1)), ref.std(dim=(0, 1))
    return (src - s_mu) / (s_std + eps) * r_std + r_mu


def restore_with_medium(gt: Tensor, rgb_object: Tensor, rgb_clear: Tensor, rgb_medium: Tensor, min_atten: float = 0.05) -> Tensor:
    """Inverts the image formation model for a *real* training image using the model's medium at that view:
    J_ref = (I - B) / A. Where the attenuation is tiny (far range / empty) the inversion is ill-conditioned and the
    rendered restored image is used instead."""
    A = effective_attenuation(rgb_object, rgb_clear)
    J = (gt - rgb_medium) / A.clamp_min(min_atten)
    return torch.where(A > min_atten, J.clamp_min(0.0), rgb_clear)


def prepare_fixer_input(
    rgb: Tensor,
    rgb_object: Tensor,
    rgb_clear: Tensor,
    rgb_medium: Tensor,
    fix_space: str = "clear",
    ref_gt: Optional[Tensor] = None,
    ref_render: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Builds the [0, 1] image (and optional reference) handed to the fixer.

    Returns (fixer_input, fixer_reference_or_None, scale) where ``scale`` maps the normalised restored image back
    to radiance (1 in 'underwater' space).
    """
    if fix_space == "underwater":
        return rgb.clamp(0, 1), (ref_gt.clamp(0, 1) if ref_gt is not None else None), rgb.new_tensor(1.0)
    if fix_space != "clear":
        raise ValueError(f"unknown fix_space {fix_space}")
    s = exposure_scale(rgb_clear)
    inp = (rgb_clear / s).clamp(0, 1)
    ref = None
    if ref_gt is not None:
        if ref_render is not None:  # medium-decoupled reference: restore the real training image
            r_obj, r_clear, r_med = ref_render
            ref = (restore_with_medium(ref_gt, r_obj, r_clear, r_med) / s).clamp(0, 1)
        else:
            ref = ref_gt.clamp(0, 1)
    return inp, ref, s


def compose_target(
    fixed: Tensor,
    rgb: Tensor,
    rgb_object: Tensor,
    rgb_clear: Tensor,
    rgb_medium: Tensor,
    scale: Tensor,
    fix_space: str = "clear",
    do_color_match: bool = True,
) -> Tensor:
    """Turns the fixer output into an underwater pseudo-target consistent with the learned medium.

    'clear'      : I_tgt = A * (Fix(J/s) * s) + B   (saturated pixels of J/s keep the rendered J)
    'underwater' : I_tgt = Fix(I)                    (plain Difix3D-style baseline, for ablation)
    """
    if fix_space == "underwater":
        out = color_match(fixed, rgb) if do_color_match else fixed
        return out.clamp(0, 1)
    inp = rgb_clear / scale
    J_fix = fixed * scale
    J_fix = torch.where(inp >= 0.999, rgb_clear, J_fix)  # clipped highlights carry no information for the fixer
    if do_color_match:
        J_fix = color_match(J_fix, rgb_clear).clamp_min(0.0)
    A = effective_attenuation(rgb_object, rgb_clear)
    return (A * J_fix + rgb_medium).clamp(0, 1)


def resize_hw(img: Tensor, h: int, w: int) -> Tensor:
    """Bilinear resize of an (H, W, C) image."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return F.interpolate(img.permute(2, 0, 1)[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


# --------------------------------------------------------------------------------------------------------------------
# Fixers
# --------------------------------------------------------------------------------------------------------------------


class IdentityFixer:
    """Returns its input: pseudo-views then only self-distil the current model (ablation baseline)."""

    uses_reference = False

    def __call__(self, images: Sequence[Tensor], refs: Optional[Sequence[Optional[Tensor]]] = None) -> List[Tensor]:
        return [img.clone() for img in images]

    def offload(self) -> None:
        pass


class DifixFixer:
    """Single-step diffusion fixer Difix / Difix-ref (Wu et al., "Difix3D+", CVPR 2025).

    Requires the Difix3D repository (for ``pipeline_difix.py``) and its pinned dependencies, see
    ``scripts/setup_difix.sh``. The pipeline is loaded lazily, kept on the CPU between refresh rounds and moved to
    the GPU only while fixing, so it does not compete with the Gaussians for GPU memory during training.
    """

    def __init__(
        self,
        model_id: str = "nvidia/difix_ref",
        use_reference: bool = True,
        difix_src: str = "",
        fp16: bool = True,
        max_side: int = 768,
        prompt: str = "remove degradation",
        timestep: int = 199,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.uses_reference = use_reference
        self.difix_src = difix_src or os.environ.get("DIFIX_SRC", "")
        self.fp16 = fp16
        self.max_side = max_side
        self.prompt = prompt
        self.timestep = timestep
        self.device = device
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        if self.difix_src and self.difix_src not in sys.path:
            sys.path.insert(0, self.difix_src)
        try:
            from pipeline_difix import DifixPipeline  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Could not import Difix's `pipeline_difix`. Clone https://github.com/nv-tlabs/Difix3D, install its "
                "requirements (see scripts/setup_difix.sh) and pass --pipeline.model.pseudo-difix-src "
                "/path/to/Difix3D/src (or set DIFIX_SRC)."
            ) from e
        kwargs = {"trust_remote_code": True}
        if self.fp16:
            kwargs["torch_dtype"] = torch.float16
        pipe = DifixPipeline.from_pretrained(self.model_id, **kwargs)
        if getattr(pipe, "safety_checker", None) is not None:
            pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def offload(self) -> None:
        if self.pipe is not None:
            self.pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _to_pil(img: Tensor, h: int, w: int):
        from PIL import Image

        arr = (resize_hw(img.float().cpu(), h, w).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
        return Image.fromarray(arr)

    def _fix_one(self, img: Tensor, ref: Optional[Tensor], max_side: int) -> Tensor:
        import numpy as np

        H, W = img.shape[:2]
        s = min(1.0, max_side / max(H, W))
        h, w = max(8, int(round(H * s / 8)) * 8), max(8, int(round(W * s / 8)) * 8)
        call = dict(
            prompt=self.prompt,
            image=self._to_pil(img, h, w),
            height=h,
            width=w,
            num_inference_steps=1,
            timesteps=[self.timestep],
            guidance_scale=0.0,
        )
        if ref is not None:
            call["ref_image"] = self._to_pil(ref, h, w)
        pil = self.pipe(**call).images[0]
        out = torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0)
        return resize_hw(out, H, W).to(img.device, img.dtype)

    @torch.no_grad()
    def __call__(self, images: Sequence[Tensor], refs: Optional[Sequence[Optional[Tensor]]] = None) -> List[Tensor]:
        """Fixes every image. The fixer shares the GPU with the Gaussians being trained, so on CUDA out-of-memory it
        frees the cache and retries at a smaller resolution (x0.75 down to 384 px); if that still fails the image is
        returned unchanged (its pseudo-target then equals the current render). Training is never interrupted."""
        self._load()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release memory cached by the training renders before loading the fixer
        try:
            self.pipe.to(self.device)
        except torch.cuda.OutOfMemoryError:
            self.offload()
            print(f"[MDGR] fixer could not be moved to {self.device} (out of memory): pseudo-views left unfixed this round")
            return [img.clone() for img in images]
        outs, n_retry, n_skip = [], 0, 0
        try:
            for k, img in enumerate(images):
                ref = refs[k] if (refs is not None and self.uses_reference) else None
                side, out = self.max_side, None
                while out is None:
                    try:
                        out = self._fix_one(img, ref, side)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        side = int(side * 0.75)
                        if side < 384:
                            out = img.clone()
                            n_skip += 1
                        else:
                            n_retry += 1
                outs.append(out)
        finally:
            self.offload()
        if n_retry or n_skip:
            print(f"[MDGR] fixer out of memory: {n_retry} retries at lower resolution, {n_skip}/{len(images)} views left unfixed")
        return outs


def make_fixer(name: str, model_id: str = "", difix_src: str = "", fp16: bool = True, max_side: int = 768):
    if name == "identity":
        return IdentityFixer()
    if name in ("difix", "difix_ref"):
        use_ref = name == "difix_ref"
        return DifixFixer(
            model_id=model_id or ("nvidia/difix_ref" if use_ref else "nvidia/difix"),
            use_reference=use_ref,
            difix_src=difix_src,
            fp16=fp16,
            max_side=max_side,
        )
    raise ValueError(f"unknown pseudo-view fixer {name}")
