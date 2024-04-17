#!/bin/bash -e   
#SBATCH --job-name=EMRI_training   # job name (shows up in the queue)
#SBATCH --time=00-00:10:00  # Walltime (DD-HH:MM:SS)
#SBATCH --gpus-per-node=A100:1    # GPU resources required per node e.g. A100:1
#SBATCH --cpus-per-task=2   # number of CPUs per task (1 by default)
#SBATCH --mem=32G         # amount of memory per node (1 by default)
##SBATCH --output=./slurm_outputs/slurm-%j.out

##SBATCH --qos=debug          # debug QOS for high priority job tests


# load required modules and environments
module purge
module load Miniconda3
module load cuDNN/8.6.0.163-CUDA-11.8.0
export PYTHONNOUSERSITE=1
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate few_env_py3_10

# run the training script
python plot_reconstructions.py