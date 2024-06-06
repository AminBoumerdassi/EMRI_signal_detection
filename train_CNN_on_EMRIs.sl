#!/bin/bash -e   
#SBATCH --job-name=EMRI_training   # job name (shows up in the queue)
#SBATCH --time=00-03:00:00  # Walltime (DD-HH:MM:SS)
#SBATCH --partition=skylake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2   # number of CPUs per task (1 by default)
#SBATCH --mem=32G         # amount of memory per node (1 by default)
#SBATCH --output=./slurm_training_outputs/slurm-%j.out

# load required modules and environments
module purge
module load mamba
module load cuda/11.7.0
module load cudnn/8.4.1.50-cuda-11.7.0
conda activate few_env_ozstar

# run the training script
python train_CNN_on_EMRIs.py