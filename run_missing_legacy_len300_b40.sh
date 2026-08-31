#!/bin/bash
#SBATCH -A naiss2025-5-659-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --array=0-5
#SBATCH -o ./srun_logs/missing_legacy_len300_b40_%A_%a.out
#SBATCH -e ./srun_logs/missing_legacy_len300_b40_%A_%a.err

set -euo pipefail

PROJECT_DIR=/home/${USER}/CSDI_social-interaction
PYTHON=/home/${USER}/csdi_env/bin/python
BASE_CONFIG=config/base_scenmap.yaml

DATA_LENGTH=300
NSAMPLE=30
EPOCHS=30
ITR_PER_EPOCH=500
BATCH_SIZE=40
MAX_NEIGHBORS=8

POINTWISE_COLLISION_WEIGHT=0.02
CLEARANCE_LOSS_WEIGHT=0.02
PATH_COLLISION_LOSS_WEIGHT=0.02
SOCIAL_COLLISION_WEIGHT=0.02
OBSTACLE_CLEARANCE_WEIGHT=1.0
OBSTACLE_CLEARANCE_MARGIN=0.01
SOCIAL_MARGIN=0.003

# The six legacy experiments that timed out before producing complete results.
TARGET_STRATEGIES=("know_first" "know_first" "know_first" "random" "random" "random")
SCENARIOS=("3-1" "3-1" "4-1" "3-1" "3-1" "4-1")
VARIANTS=("obs_loss" "full" "full" "obs_loss" "full" "full")

TARGET_STRATEGY=${TARGET_STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

module purge
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb

[[ -x "$PYTHON" ]] || { echo "Missing GPU Python environment: $PYTHON" >&2; exit 1; }
cd "$PROJECT_DIR"
[[ -d data/simulation_data ]] || { echo "Missing dataset: $PROJECT_DIR/data/simulation_data" >&2; exit 1; }
mkdir -p srun_logs config/generated_ablation

export PYTHONPATH=/home/${USER}
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

CFG_NAME="generated_ablation/missing_legacy_${TARGET_STRATEGY}_len300_b40_${SCENARIO}_${VARIANT}_e${EPOCHS}_len${DATA_LENGTH}_n${NSAMPLE}.yaml"

"$PYTHON" - <<PY
import yaml
from pathlib import Path

base_cfg = Path('${BASE_CONFIG}')
out_cfg = Path('config/${CFG_NAME}')

with open(base_cfg, 'r') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('dataset', {})
cfg['dataset']['scenarios'] = ['${SCENARIO}']
cfg['dataset']['missing_strategy'] = 'know_first'
cfg['dataset']['missing_ratio'] = 0.5

cfg.setdefault('train', {})
cfg['train']['epochs'] = ${EPOCHS}
cfg['train']['itr_per_epoch'] = ${ITR_PER_EPOCH}
cfg['train']['batch_size'] = ${BATCH_SIZE}

cfg.setdefault('valid', {})
cfg.setdefault('test', {})
cfg['valid']['batch_size'] = ${BATCH_SIZE}
cfg['test']['batch_size'] = ${BATCH_SIZE}

cfg.setdefault('model', {})
cfg['model']['target_strategy'] = '${TARGET_STRATEGY}'
cfg['model']['scene_goal_channels'] = 5
cfg['model']['socialemb'] = 64
cfg['model']['social_hidden'] = 64
cfg['model']['social_hidden_dim'] = 64
cfg['model']['fusionemb'] = cfg['model'].get('scenmapemb', 256)
cfg['model']['collision_loss_weight'] = ${POINTWISE_COLLISION_WEIGHT}
cfg['model']['clearance_loss_weight'] = ${CLEARANCE_LOSS_WEIGHT}
cfg['model']['path_collision_loss_weight'] = ${PATH_COLLISION_LOSS_WEIGHT}
cfg['model']['social_collision_loss_weight'] = ${SOCIAL_COLLISION_WEIGHT}
cfg['model']['obstacle_clearance_weight'] = ${OBSTACLE_CLEARANCE_WEIGHT}
cfg['model']['obstacle_clearance_margin'] = ${OBSTACLE_CLEARANCE_MARGIN}
cfg['model']['social_margin'] = ${SOCIAL_MARGIN}

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f'wrote {out_cfg}')
PY

EXTRA_ARGS=()
if [[ "$TARGET_STRATEGY" == "random" ]]; then
    EXTRA_ARGS+=(--force_random_target_know_first)
fi

echo "Running missing legacy ${TARGET_STRATEGY} ${VARIANT} on ${SCENARIO}: $(date)"

"$PYTHON" exe_simulation_scenmap.py \
    --config "$CFG_NAME" \
    --device cuda:0 \
    --data_length "$DATA_LENGTH" \
    --nsample "$NSAMPLE" \
    --model_variant "$VARIANT" \
    --ablation_order legacy \
    --max_neighbors "$MAX_NEIGHBORS" \
    --eval_collision \
    --scenarios "$SCENARIO" \
    "${EXTRA_ARGS[@]}"

echo "Finished missing legacy ${TARGET_STRATEGY} ${VARIANT} on ${SCENARIO}: $(date)"
