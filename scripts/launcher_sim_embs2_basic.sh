#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -p fasse_gpu
#SBATCH -t 4:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-9
#SBATCH -o ./slurm/sim-embs-basic.%a.out

#
source ~/.bashrc
conda activate cuda116
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method pca &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method tsne &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method cvae &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method crae &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method resnet &
python train_sim_embs.py --silent --sim $SLURM_ARRAY_TASK_ID --task basic --method unet &
wait
