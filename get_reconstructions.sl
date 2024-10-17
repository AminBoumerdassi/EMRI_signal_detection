#!/bin/bash -e   
#SBATCH --job-name=get_EMRI_reconstructions   # job name (shows up in the queue)
#SBATCH --time=00-00:10:00  # Walltime (DD-HH:MM:SS)
#SBATCH --partition=skylake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1   # number of CPUs per task (1 by default)
#SBATCH --mem=3G         # amount of memory per node (1 by default)
#SBATCH --output=./slurm_testing_outputs/slurm-%j.out

# load required modules and environments
source init_env.sh

# run the training script
python get_reconstructions.py