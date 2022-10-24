#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -p fasse_gpu
#SBATCH -t 1:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-4
#SBATCH -o ./slurm/sim-effs2.%a.out

#
source ~/.bashrc
conda activate cuda116


# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method pca &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method tsne &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method crae &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method cvae &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method unet &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method resnet &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method local &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method avg &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method wx &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method resnet_sup &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method unet_sup &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method unet_sup_car &
# python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID)) --method car &
wait
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method pca &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method tsne &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method crae &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method cvae &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method unet &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method resnet &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method local &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method avg &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method wx &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method resnet_sup &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method unet_sup &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method unet_sup_car &
python train_sim_effects.py --silent --embsdir results-sim2 --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --method car &
wait
