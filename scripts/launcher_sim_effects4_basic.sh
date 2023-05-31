#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -p fasse_gpu
#SBATCH -t 0:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-9
#SBATCH -o ./slurm/sim-effs.%a.out

#
source ~/.bashrc
conda activate cuda116
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method pca &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method tsne &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method crae &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method cvae &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method unet &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method resnet &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method local &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method avg &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method wx &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method resnet_sup &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method unet_sup &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method unet_sup_car &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID output --method car &
wait
