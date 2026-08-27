#!/bin/bash
#SBATCH -A naiss2025-5-659
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 24:00:00
#SBATCH -o ./srun_logs/batch40_baseline_randomtarget_knowfirst_s1_%j.out
#SBATCH -e ./srun_logs/batch40_baseline_randomtarget_knowfirst_s1_%j.err

set -euo pipefail

PROJECT_DIR=/cephyr/users/haoxian/Alvis/CSDI_baseline_test/CSDI
PYTHON=/cephyr/users/haoxian/Alvis/csdi_env/bin/python
BASE_CONFIG=config/base_scenmap.yaml

DATA_LENGTH=100
NSAMPLE=30
EPOCHS=30
ITR_PER_EPOCH=500
BATCH_SIZE=40
MAX_NEIGHBORS=8

SCENARIO="1-1"
VARIANT="baseline"

module purge
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0

cd "$PROJECT_DIR"
mkdir -p srun_logs
mkdir -p config/generated_ablation

export PYTHONPATH=/cephyr/users/haoxian/Alvis/CSDI_baseline_test

CFG_NAME="generated_ablation/batch40_randomtarget_knowfirst_${SCENARIO}_${VARIANT}_e${EPOCHS}_len${DATA_LENGTH}_n${NSAMPLE}.yaml"

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
cfg['model']['target_strategy'] = 'random'
cfg['model']['scene_goal_channels'] = 5
cfg['model']['socialemb'] = 64
cfg['model']['social_hidden'] = 64
cfg['model']['social_hidden_dim'] = 64
cfg['model']['fusionemb'] = cfg['model'].get('scenmapemb', 256)
cfg['model']['collision_loss_weight'] = 0.02
cfg['model']['social_collision_loss_weight'] = 0.02
cfg['model']['social_margin'] = 0.003

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f'wrote {out_cfg}')
PY

echo "Running batch40 baseline random-target know-first on scenario ${SCENARIO}"

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

echo "Finished batch40 baseline random-target know-first on ${SCENARIO}: $(date)"
