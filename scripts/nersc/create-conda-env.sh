#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --account=m3504_g
#SBATCH --qos=regular
#SBATCH --time-min=00:30:00
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --job-name=create-conda-env
#SBATCH --output=logs/%j-out.out
#SBATCH --error=logs/%j-err.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=begin,end,fail

module load python
nvidia-smi
export SLURM_CPU_BIND="cores"
# cd to env dir
cd ../../environment
# create conda env
conda env create -f env_train_gpu.yaml -n train_gpu
# activate env
conda activate train_gpu
echo "conda env created and activated"
conda env list

