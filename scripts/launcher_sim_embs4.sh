#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -p fasse_gpu
#SBATCH -t 7:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-4
#SBATCH -o ./slurm/sim-embs3.%a.out

#
source ~/.bashrc
conda activate cuda116

python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method pca &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method tsne &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method cvae &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method crae &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method resnet &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID)) --output results-sim6 --d 6 --method unet &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method pca &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method tsne &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method cvae &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method crae &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method resnet &
python train_sim_embs.py --silent --sim $((2*SLURM_ARRAY_TASK_ID + 1)) --output results-sim6 --d 6 --method unet &
wait
