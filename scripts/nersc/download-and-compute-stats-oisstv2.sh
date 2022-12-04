#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --account=m3504
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --time-min=00:25:00
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=230GB
#SBATCH --job-name=oisstv2-download-compute-stats
#SBATCH --output=logs/oisstv2-download-compute-stats-%j.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=begin,end,fail

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

srun python download_oisstv2.py "$OISSTV2_DIR"
echo "Proceeding to process OISSTv2 data..."
srun python oisstv2_boxed_data_preprocessing.py "$OISSTV2_DIR"
