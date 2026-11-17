#!/usr/bin/env bash
# Installs the Difix fixer (Wu et al., "Difix3D+", CVPR 2025) into the existing WaterSplatting environment.
# Difix pins versions that still support Python 3.8 / torch 2.1, so it can live in the same env as nerfstudio 1.1.4.
#   usage: bash scripts/setup_difix.sh [install_dir]      then: export DIFIX_SRC=<install_dir>/Difix3D/src
# Note: Difix code and weights are released under the NVIDIA license (non-commercial research use) - check
# https://github.com/nv-tlabs/Difix3D/blob/main/LICENSE.txt before use.
set -euo pipefail
DIR="${1:-$(pwd)/third_party}"
mkdir -p "$DIR"
if [ ! -d "$DIR/Difix3D" ]; then
  git clone --depth 1 https://github.com/nv-tlabs/Difix3D.git "$DIR/Difix3D"
fi
# Do NOT reinstall torch: only the pinned diffusion stack from Difix3D/requirements.txt.
pip install "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" "huggingface-hub==0.25.1" einops
echo
echo "Difix installed. Run:  export DIFIX_SRC=$DIR/Difix3D/src"
echo "Weights (nvidia/difix_ref, nvidia/difix) are downloaded from HuggingFace on first use."
