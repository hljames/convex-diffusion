#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --account=m3504_g
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --time-min=03:30:00
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=85GB
#SBATCH --job-name=oisst-unet
#SBATCH --output=logs/oisstv2-unet-%j.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=fail
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

#    --licenses=scratch,cfs

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# Load python
module load python
source activate eval_cpu

cd ../..
# run the script with all extra arguments
srun python run.py "$@" \
  experiment=oisst_unet \
  trainer=ddp \
