#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --account=m3504_g
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --time-min=03:00:00
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --job-name=oisst-resume-small
#SBATCH --output=logs/oisstv2-resume-small-%j.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=fail
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue

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
  trainer.gpus=-1 \
  datamodule.data_dir="/global/cfs/cdirs/m3504/" \
  work_dir="./results" ckpt_dir="./results/checkpoints/" log_dir="./results/logs/" \
  callbacks.model_checkpoint.dirpath="./results/checkpoints/" \
 # ++trainer.strategy="ddp_find_unused_parameters_false" \
