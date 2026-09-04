#!/bin/bash
#补跑超时且没有完整结果的 7 个实验组。
#SBATCH -A naiss2025-5-659-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --array=0-6
#SBATCH -o ./srun_logs/missing_ablation_len300_b40_%A_%a.out
#SBATCH -e ./srun_logs/missing_ablation_len300_b40_%A_%a.err

set -euo pipefail
PROJECT_DIR=/home/${USER}/CSDI_social-interaction
PYTHON=/home/${USER}/csdi_env/bin/python
DATA_LENGTH=300; NSAMPLE=30; EPOCHS=30; ITR_PER_EPOCH=500; BATCH_SIZE=40; MAX_NEIGHBORS=8

ORDERS=(legacy legacy legacy legacy reordered reordered reordered)
STRATEGIES=(know_first know_first random random random random know_first)
SCENARIOS=(3-1 4-1 3-1 4-1 3-1 4-1 3-1)
VARIANTS=(fusion obs_loss obs_loss obs_loss social_loss social_loss social_loss)

ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}; STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}; VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

module purge
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
cd "$PROJECT_DIR"
mkdir -p srun_logs config/generated_ablation
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -d data/simulation_data ]] || { echo "Missing dataset" >&2; exit 1; }

CFG="generated_ablation/missing_${ORDER}_${STRATEGY}_${SCENARIO}_${VARIANT}_len300_b40.yaml"
"$PYTHON" - <<PY
import yaml
from pathlib import Path
base=yaml.safe_load(open('config/base_scenmap.yaml'))
base.setdefault('dataset',{}).update(scenarios=['${SCENARIO}'], missing_strategy='know_first', missing_ratio=0.5)
base.setdefault('train',{}).update(epochs=${EPOCHS}, itr_per_epoch=${ITR_PER_EPOCH}, batch_size=${BATCH_SIZE})
base.setdefault('valid',{})['batch_size']=${BATCH_SIZE}; base.setdefault('test',{})['batch_size']=${BATCH_SIZE}
m=base.setdefault('model',{}); m.update(target_strategy='${STRATEGY}', scene_goal_channels=5, socialemb=64, social_hidden=64, social_hidden_dim=64, fusionemb=m.get('scenmapemb',256), social_collision_loss_weight=0.2, social_margin=0.04)
Path('${CFG}').parent.mkdir(parents=True,exist_ok=True); yaml.safe_dump(base,open('${CFG}','w'),sort_keys=False)
PY

EXTRA=(); [[ "$STRATEGY" == random ]] && EXTRA+=(--force_random_target_know_first)
"$PYTHON" exe_simulation_scenmap.py --config "$CFG" --device cuda:0 --data_length "$DATA_LENGTH" --nsample "$NSAMPLE" --model_variant "$VARIANT" --ablation_order "$ORDER" --max_neighbors "$MAX_NEIGHBORS" --eval_collision --scenarios "$SCENARIO" "${EXTRA[@]}"
