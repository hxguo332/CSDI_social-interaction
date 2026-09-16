#!/bin/bash
# Three short loss-weight pilots on legacy/know-first/full, scenario 3-1.
#SBATCH -A naiss2025-5-659-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -o ./srun_logs/weight_pilot_legacy_knowfirst_3-1_%A_%a.out
#SBATCH -e ./srun_logs/weight_pilot_legacy_knowfirst_3-1_%A_%a.err

set -euo pipefail
PROJECT_DIR=/home/${USER}/CSDI_social-interaction
PYTHON=/home/${USER}/csdi_env/bin/python

LABELS=(A B C)
SOCIAL_WEIGHTS=(0.05 0.10 0.05)
CLEARANCE_WEIGHTS=(0.10 0.10 0.20)
PATH_WEIGHTS=(0.10 0.10 0.20)

ID=${SLURM_ARRAY_TASK_ID}
LABEL=${LABELS[$ID]}
SOCIAL_WEIGHT=${SOCIAL_WEIGHTS[$ID]}
CLEARANCE_WEIGHT=${CLEARANCE_WEIGHTS[$ID]}
PATH_WEIGHT=${PATH_WEIGHTS[$ID]}
CFG="generated_ablation/weight_pilot_${LABEL}_legacy_know_first_3-1_full_len300_b40.yaml"

module purge
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
cd "$PROJECT_DIR"
mkdir -p srun_logs config/generated_ablation
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -d data/simulation_data ]] || { echo "Missing dataset" >&2; exit 1; }

"$PYTHON" - <<PY
import yaml
from pathlib import Path

config = yaml.safe_load(open('config/base_scenmap.yaml'))
config.setdefault('dataset', {}).update(scenarios=['3-1'], missing_strategy='know_first', missing_ratio=0.5)
config.setdefault('train', {}).update(epochs=5, itr_per_epoch=100, batch_size=40)
config.setdefault('valid', {})['batch_size'] = 40
config.setdefault('test', {})['batch_size'] = 40
config.setdefault('experiment', {})['weight_pilot'] = '${LABEL}'
config.setdefault('model', {}).update(
    target_strategy='know_first',
    scene_goal_channels=5,
    socialemb=64,
    social_hidden=64,
    social_hidden_dim=64,
    fusionemb=config['model'].get('scenmapemb', 256),
    collision_loss_weight=0.5,
    clearance_loss_weight=${CLEARANCE_WEIGHT},
    path_collision_loss_weight=${PATH_WEIGHT},
    social_collision_loss_weight=${SOCIAL_WEIGHT},
    obstacle_clearance_weight=1.0,
    obstacle_clearance_margin=0.01,
    social_margin=0.04,
)
path = Path('config') / '${CFG}'
path.parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(config, open(path, 'w'), sort_keys=False)
PY

echo "Pilot ${LABEL}: social=${SOCIAL_WEIGHT}, clearance=${CLEARANCE_WEIGHT}, path=${PATH_WEIGHT}"
"$PYTHON" exe_simulation_scenmap.py \
    --config "$CFG" \
    --device cuda:0 \
    --data_length 300 \
    --nsample 30 \
    --model_variant full \
    --ablation_order legacy \
    --max_neighbors 8 \
    --eval_collision \
    --scenarios 3-1
