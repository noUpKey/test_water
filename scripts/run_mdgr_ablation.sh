#!/usr/bin/env bash
# MDGR ablation on one scene (same split / seed protocol as the reproduction):
#   baseline | self-distillation (identity fixer) | Difix in underwater space | MDGR (Difix-ref in restored space)
#
#   usage: bash scripts/run_mdgr_ablation.sh <undistorted_scene_dir> <scene_name> [seed] [out_dir]
#   requires: pip install --no-use-pep517 -e .   and   bash scripts/setup_difix.sh  (export DIFIX_SRC=...)
set -euo pipefail
DATA="$1"; SCENE="$2"; SEED="${3:-42}"; OUT="${4:-outputs}"
DATA_ARGS=(colmap --downscale-factor 1 --colmap-path sparse --data "$DATA" --images-path images --eval-mode interval --eval-interval 8)

run () {  # run <tag> <method> [extra model args...]
  local tag="$1" method="$2"; shift 2
  local ts="${tag}_seed${SEED}"
  ns-train "$method" --vis tensorboard --output-dir "$OUT" --experiment-name "$SCENE" --timestamp "$ts" \
      --machine.seed "$SEED" "$@" "${DATA_ARGS[@]}"
  local run_dir="$OUT/$SCENE/$method/$ts"
  ns-eval --load-config "$run_dir/config.yml" --output-path "$run_dir/eval.json" --render-output-path "$run_dir/renders"
}

run baseline  water-splatting
run selfdist  water-splatting-pseudo --pipeline.model.pseudo-fixer identity
run difix_uw  water-splatting-pseudo --pipeline.model.pseudo-fixer difix_ref --pipeline.model.pseudo-fix-space underwater
run mdgr      water-splatting-pseudo --pipeline.model.pseudo-fixer difix_ref --pipeline.model.pseudo-fix-space clear \
              --pipeline.model.pseudo-debug-dir "$OUT/$SCENE/mdgr_debug_seed${SEED}"

python scripts/radial_metrics.py "$OUT/$SCENE"/water-splatting/baseline_seed${SEED}/renders \
    "$OUT/$SCENE"/water-splatting-pseudo/*_seed${SEED}/renders
