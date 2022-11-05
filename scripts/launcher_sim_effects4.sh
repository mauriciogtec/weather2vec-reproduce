#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -p fasse_gpu
#SBATCH -t 1:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-4
#SBATCH -o ./slurm/sim-effs4d.%a.out

#
source ~/.bashrc
conda activate cuda116


for sparse in "" "--sparse"
do
    for task in "nonlinear" "basic"
    do
        for i in 0 1
        do
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method pca &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method tsne &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method crae &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method cvae &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method unet &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method resnet &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method local &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method avg &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method causal_wx &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method resnet_sup &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method unet_sup &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method unet_sup_car &
            python train_sim_effects.py $sparse --task=$task --embsdir results-sim6 --silent --sim $((2*SLURM_ARRAY_TASK_ID + i)) --method car &
            wait
        done
    done
done