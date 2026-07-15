
#!/bin/bash

set -euo pipefail



cd /cephyr/users/haoxian/Alvis/CSDI_baseline_test/CSDI



module purge

module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0



export PYTHONPATH=/cephyr/users/haoxian/Alvis/CSDI_baseline_test

export MPLCONFIGDIR=/tmp/$USER/matplotlib

mkdir -p "$MPLCONFIGDIR"



PYTHON=/cephyr/users/haoxian/Alvis/csdi_env/bin/python



rm -rf save/simulation_random_target_knowfirst_baseline_1-1_20260703_160224/near_collision_case_plots

rm -rf save/simulation_random_target_knowfirst_baseline_2-3_20260703_160225/near_collision_case_plots

rm -rf save/simulation_random_target_knowfirst_baseline_3-1_20260703_160226/near_collision_case_plots

rm -rf save/simulation_random_target_knowfirst_baseline_4-1_20260703_160232/near_collision_case_plots

rm -rf save/simulation_task_1-1_20260627_011940/near_collision_case_plots

rm -rf save/simulation_task_2-3_20260627_012200/near_collision_case_plots

rm -rf save/simulation_task_3-1_20260627_015142/near_collision_case_plots

rm -rf save/simulation_task_4-1_20260627_014709/near_collision_case_plots



$PYTHON visualize_near_collision_cases.py --pickle save/simulation_random_target_knowfirst_baseline_1-1_20260703_160224/generated_outputs_nsample30_nonaug_baseline.pk --outdir save/simulation_random_target_knowfirst_baseline_1-1_20260703_160224/near_collision_case_plots --scenario 1-1 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_random_target_knowfirst_baseline_2-3_20260703_160225/generated_outputs_nsample30_nonaug_baseline.pk --outdir save/simulation_random_target_knowfirst_baseline_2-3_20260703_160225/near_collision_case_plots --scenario 2-3 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_random_target_knowfirst_baseline_3-1_20260703_160226/generated_outputs_nsample30_nonaug_baseline.pk --outdir save/simulation_random_target_knowfirst_baseline_3-1_20260703_160226/near_collision_case_plots --scenario 3-1 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_random_target_knowfirst_baseline_4-1_20260703_160232/generated_outputs_nsample30_nonaug_baseline.pk --outdir save/simulation_random_target_knowfirst_baseline_4-1_20260703_160232/near_collision_case_plots --scenario 4-1 --num_cases 5 --max_neighbors 8



$PYTHON visualize_near_collision_cases.py --pickle save/simulation_task_1-1_20260627_011940/generated_outputs_nsample30_nonaug_task.pk --outdir save/simulation_task_1-1_20260627_011940/near_collision_case_plots --scenario 1-1 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_task_2-3_20260627_012200/generated_outputs_nsample30_nonaug_task.pk --outdir save/simulation_task_2-3_20260627_012200/near_collision_case_plots --scenario 2-3 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_task_3-1_20260627_015142/generated_outputs_nsample30_nonaug_task.pk --outdir save/simulation_task_3-1_20260627_015142/near_collision_case_plots --scenario 3-1 --num_cases 5 --max_neighbors 8

$PYTHON visualize_near_collision_cases.py --pickle save/simulation_task_4-1_20260627_014709/generated_outputs_nsample30_nonaug_task.pk --outdir save/simulation_task_4-1_20260627_014709/near_collision_case_plots --scenario 4-1 --num_cases 5 --max_neighbors 8

