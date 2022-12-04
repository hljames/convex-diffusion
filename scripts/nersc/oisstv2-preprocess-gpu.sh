#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --account=m3504_g
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --time-min=06:00:00
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem=155GB
#SBATCH --job-name=oisst-preprocess
#SBATCH --output=logs/oisstv2-preprocess-%j.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=begin,end,fail
#    --licenses=scratch,cfs

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true
# Load python
module load python
cd ../..
# pip install -r requirements.txt --user
source activate eval_cpu

DATA_DIR=$CFS/m3504
OISSTV2_DIR="$DATA_DIR"/oisstv2-daily
mkdir -p "$OISSTV2_DIR"
cd data/dataset_creation


# srun python oisstv2_boxed_data_preprocessing.py "$OISSTV2_DIR"
srun python oisstv2_boxed_data_preprocessing.py "$OISSTV2_DIR" True
