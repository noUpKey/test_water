<p align="center">
    <h1>
        <span class="title-main"><span>WaterSplatting</span></span>
        <span class="title-small">Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting</span>
      </h1>
  <p align="center">
    <a href="https://github.com/IronLu11">Huapeng Li</a>
    ·
    <a href="https://www.linkedin.com/in/wenxuan-song-901123188/">Wenxuan Song</a>
    ·
    <a href="https://www.linkedin.com/in/tianao-xu-066b51258/">Tianao Xu</a>
    ·
    <a href="https://www.linkedin.com/in/alexandre-elsig-516642116/">Alexandre Elsig</a>
    ·
    <a href="https://jkulhanek.com/">Jonas Kulhanek</a>
  </p><p align="center">
    <b><i>3DV 2025</i></b>
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2408.08206">📄 Paper</a> | <a href="https://water-splatting.github.io/">🌐 Project Page</a></h3>
  <div align="center"></div>
</p>
<br/>
<p align="center">
  <img alt="WaterSplatting Reconstruction" src=".assets/curasao.webp" />
</p>
    <p align="center" class="justify" style="font-size: 1rem;margin: 0 0 0.4rem 0; text-align-last: center">
    <strong>WaterSplatting</strong> combines 3DGS with volume rendering to enable water/fog modeling</strong>
    </p>
<p align="justify">
We introduce WaterSplatting, a novel approach that fuses volumetric rendering with 3DGS to handle underwater data effectively. 
Our method employs 3DGS for explicit geometry representation and a separate volumetric field (queried once per pixel) for capturing the scattering medium. 
This dual representation further allows the restoration of the scenes by removing the scattering medium. 
Our method outperforms state-of-the-art NeRF-based methods in rendering quality on the underwater SeaThru-NeRF dataset. 
Furthermore, it does so while offering real-time rendering performance.
</p>
<br>

## Extensions in this fork

This fork adds two optional extensions on top of the official code. Both are **off by default**, so
`ns-train water-splatting` reproduces the original method exactly.

### 1. Medium-decoupled generative refinement (MDGR): `ns-train water-splatting-pseudo`

**Motivation.** Measured on the reproduction renders (`scripts/radial_metrics.py`), the test error is concentrated at the image borders.
That is where few of the 15–25 training cameras overlap. The water itself is modelled well.

| scene (seed 42, T4) | centre | ring 2 | ring 3 | corner |
|---|---|---|---|---|
| IUI3-RedSea | 31.01 | 30.92 | 27.63 | **22.46** |
| Curaçao | 32.28 | 33.01 | 30.38 | **26.97** |
| J.G. RedSea | 25.85 | 27.74 | 22.40 | **18.93** |
| Panama | 33.79 | 31.20 | 31.26 | 31.37 |

(PSNR in dB, concentric rings from the image centre to the corners.)

**Method.** Every `pseudo_every` steps:

1. Sample pseudo-camera poses only from the *training* trajectory: interpolation between neighbouring cameras, plus small yaw/pitch, in-plane and backward perturbations.
   Stepping backward brings the poorly observed training-view borders into the frame. No test poses or test images are used.
2. Render each pseudo-view decomposed as `I = A·J + B`.
   `J` is the restored (medium-free) image, `B` the backscatter, and `A = J_att / J` the effective per-pixel attenuation.
3. Repair `J`, not `I`, with the single-step diffusion fixer [Difix](https://github.com/nv-tlabs/Difix3D) (Wu et al., CVPR 2025).
   Difix is trained on in-air images, so the restored image is in-distribution for it, while the hazy underwater render is not.
   For the reference-guided `difix_ref`, the reference is the nearest training image restored with the model's own medium, `(I_gt − B)/A`.
4. Re-apply the learned medium to get the pseudo-target `A·Fix(J) + B`.
   It is consistent with the water model by construction: an identity fixer reproduces the render exactly (`tests/test_pseudo_views.py`).
5. Distil the targets with an L1 + D-SSIM loss on one random pseudo-view per step (probability `pseudo_prob`, weight `pseudo_loss_weight`).
   The medium field gets no gradient from pseudo-views, and pseudo-views do not drive densification.

Code: [`water_splatting/pseudo_views.py`](water_splatting/pseudo_views.py) and the `MDGR` section of [`water_splatting/water_splatting.py`](water_splatting/water_splatting.py).

```bash
bash scripts/setup_difix.sh && export DIFIX_SRC=$(pwd)/third_party/Difix3D/src   # Difix pins py3.8-compatible deps
ns-train water-splatting-pseudo --vis tensorboard --machine.seed 42 \
    --pipeline.model.pseudo-debug-dir outputs/mdgr_debug \
    colmap --downscale-factor 1 --colmap-path sparse --data <scene> --images-path images --eval-mode interval --eval-interval 8
# full ablation (baseline / identity self-distillation / Difix in underwater space / MDGR) + ring metrics:
bash scripts/run_mdgr_ablation.sh <scene> <scene_name> 42
```

Main options (`--pipeline.model.<name>`):

- `pseudo-fixer {difix_ref,difix,identity}` and `pseudo-fix-space {clear,underwater}`.
- `pseudo-start-step 7000`, `pseudo-every 2000`, `pseudo-num-views 0` (0 means 2× the number of training views).
- `pseudo-loss-weight 0.3`, `pseudo-prob 0.5`, `pseudo-downscale 2`.

On a T4, each refresh round takes roughly a minute and the per-step cost rises by about 25 %.
The Difix weights are under the NVIDIA license (non-commercial research use).

### 2. Medium-aware depth gradient: `--pipeline.model.medium-depth-grad-scale <s>`

The rendered colour depends on each Gaussian's depth through `exp(-σ_attn s_i)` and through the backscatter segments.
The original rasterizer drops `dC/ds_i`, so the water model never informs Gaussian positions.
The CUDA backward pass now computes it:

`dC/ds_i = α_i T_i [−σ_attn c_i e^(−σ_attn s_i) + σ_bs c_med e^(−σ_bs s_i)]`

It is applied with scale `s` (default 0, the original behaviour). Check it with `python scripts/check_depth_grad.py` on a GPU.

### Reproduction

[`reproduction/`](reproduction/) holds the seed-42 reproduction on a Kaggle T4 (average PSNR 29.32 vs 29.69 in the paper).
It contains the splits, per-scene `eval.json` and `config.yml`, environment dumps and training logs.

## Installation

Our method is based on [nerfstudio](https://docs.nerf.studio/index.html).

### Create environment
```bash
conda create --name water_splatting -y python=3.8
conda activate water_splatting
python -m pip install --upgrade pip
```

### Install WaterSplatting

```bash
# Install PyTorch
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install cuda-toolkit with conda
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install nerfstudio
pip install nerfstudio==1.1.4
ns-install-cli

# WaterSplatting
git clone git@github.com:water-splatting/water-splatting.git
cd water-splatting
git submodule init
git submodule update --recursive
pip install --no-use-pep517 -e .
```

## Data Preprocessing
To keep consistency across different models, we recomputed the camera intrinsic/extrinsic parameters and performed distortion corrections using [COLMAP](https://github.com/colmap/colmap)'s `image_undistorter` on [SeaThru-NeRF](https://sea-thru-nerf.github.io/) dataset:
```bash
colmap image_undistorter \
  --image_path /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/images_wb \
  --input_path /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/colmap/sparse/0 \
  --output_path /your_path_to_dataset/undistorted_seathrunerf_dataset/IUI3-RedSea \
  --output_type COLMAP
```

## Training
To start the training on the undistorted SeaThru-NeRF dataset, run the following commands:
```bash
cd /your_path_to_repo/water-splatting
ns-train water-splatting --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse --data /your_path_to_dataset/undistorted_seathrunerf_dataset/IUI3-RedSea --images-path images
```

Or, to start the training on the original [SeaThru-NeRF](https://sea-thru-nerf.github.io/) dataset, run the following commands:
```bash
cd /your_path_to_repo/water-splatting
ns-train water-splatting --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea --images-path Images_wb
```
Please note that: The training and testing splits reported in our paper are different from the default splits in nerfstudio, and are consistent with the splits used in the SeaThru-NeRF paper.

## Evaluation

```bash
cd /your_path_to_repo/water-splatting
ns-eval --load-config outputs/unnamed/water-splatting/your_timestamp/config.yml --render-output-path renders/eval
```

## Interactive viewer
To start the viewer and explore the trained models, run one of the following:
```bash
ns-viewer --load-config outputs/unnamed/water-splatting/your_timestamp/config.yml
```

## Rendering videos
To render a video on a trajectory (e.g., generated from the interactive viewer), run:
```bash
ns-render camera-path --load-config outputs/unnamed/water-splatting/your_timestamp/config.yml --camera-path-filename /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/camera_paths/your_trajectory.json --output-path renders/IUI3-RedSea/water_splatting.mp4
```

Please note that the default output quality is lossy.

## Rendering dataset
To render testing set for a checkpoint, run:
```bash
ns-render dataset --load-config outputs/unnamed/water-splatting/your_timestamp/config.yml --data /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea
```
Please note that the default output quality is lossy.
</p>
</section>

## Acknowledgements
This work was supported by the Czech Science Foundation (GACR) EXPRO (grant no. 23-07973X), and by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254).
Jonas Kulhanek acknowledges travel support from the European Union’s Horizon 2020 research and innovation programme under ELISE (grant no. 951847).

## Citation
If you find our code or paper useful, please cite:
```bibtex
@article{li2025watersplatting,
  title={{W}ater{S}platting: Fast Underwater {3D} Scene Reconstruction using Gaussian Splatting},
  author={Li, Huapeng and Song, Wenxuan and Xu, Tianao and Elsig, Alexandre and Kulhanek, Jonas},
  journal={3DV},
  year={2025}
}
```
