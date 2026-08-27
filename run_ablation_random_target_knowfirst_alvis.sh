#!/bin/bash
#SBATCH -A naiss2025-5-659
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 24:00:00
#SBATCH --array=0-1
#SBATCH -o ./srun_logs/legacy_random_target_knowfirst_%A_%a.out
#SBATCH -e ./srun_logs/legacy_random_target_knowfirst_%A_%a.err

set -euo pipefail

PROJECT_DIR=/cephyr/users/haoxian/Alvis/CSDI_baseline_test/CSDI
PYTHON=/cephyr/users/haoxian/Alvis/csdi_env/bin/python
BASE_CONFIG=config/base_scenmap.yaml

DATA_LENGTH=100
NSAMPLE=30
EPOCHS=30
ITR_PER_EPOCH=500
BATCH_SIZE=4
MAX_NEIGHBORS=8

COLLISION_WEIGHT=0.02
SOCIAL_COLLISION_WEIGHT=0.02
SOCIAL_MARGIN=0.003

SCENARIOS=("3-1" "4-1")
VARIANTS=("full")

SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

module purge
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0

cd "$PROJECT_DIR"
mkdir -p srun_logs
mkdir -p config/generated_ablation

export PYTHONPATH=/cephyr/users/haoxian/Alvis/CSDI_baseline_test

echo "============================================================"
echo "Random-target + know-first comparison job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Scenario: ${SCENARIO}"
echo "Project: ${PROJECT_DIR}"
echo "Started: $(date)"
echo "============================================================"

for VARIANT in "${VARIANTS[@]}"; do
    CFG_NAME="generated_ablation/legacy_random_target_knowfirst_${SCENARIO}_${VARIANT}_e${EPOCHS}_len${DATA_LENGTH}_n${NSAMPLE}.yaml"

    echo "------------------------------------------------------------"
    echo "Preparing random-target know-first config: ${CFG_NAME}"
    echo "Variant: ${VARIANT}"
    echo "Scenario: ${SCENARIO}"
    echo "Time: $(date)"
    echo "------------------------------------------------------------"

    "$PYTHON" - <<PY
import yaml
from pathlib import Path

base_cfg = Path('${BASE_CONFIG}')
out_cfg = Path('config/${CFG_NAME}')

with open(base_cfg, 'r') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('dataset', {})
cfg['dataset']['scenarios'] = ['${SCENARIO}']

# Random-target + know-first comparison:
# The model target strategy is random, while the trajectory completion task is future-half prediction.
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
cfg['model']['target_strategy'] = 'random'

cfg['model']['scene_goal_channels'] = 5
cfg['model']['socialemb'] = 64
cfg['model']['social_hidden'] = 64
cfg['model']['social_hidden_dim'] = 64
cfg['model']['fusionemb'] = cfg['model'].get('scenmapemb', 256)

cfg['model']['collision_loss_weight'] = ${COLLISION_WEIGHT}
cfg['model']['social_collision_loss_weight'] = ${SOCIAL_COLLISION_WEIGHT}
cfg['model']['social_margin'] = ${SOCIAL_MARGIN}

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f'wrote {out_cfg}')
PY

    echo "Running random-target know-first ${VARIANT} on scenario ${SCENARIO}"

    "$PYTHON" exe_simulation_scenmap.py \
        --config "${CFG_NAME}" \
        --device cuda:0 \
        --data_length "${DATA_LENGTH}" \
        --nsample "${NSAMPLE}" \
        --model_variant "${VARIANT}" \
        --ablation_order legacy \
        --max_neighbors "${MAX_NEIGHBORS}" \
        --eval_collision \
        --force_random_target_know_first \
        --scenarios "${SCENARIO}"

    echo "Finished random-target know-first ${VARIANT} on ${SCENARIO}: $(date)"
done

echo "============================================================"
echo "All random-target know-first variants finished for scenario ${SCENARIO}"
echo "Finished: $(date)"
echo "============================================================"
