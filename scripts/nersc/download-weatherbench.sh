#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --account=m3504_g
#SBATCH --licenses=cfs
#SBATCH --qos=regular
#SBATCH --time-min=05:00:00
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=download-weatherbench
#SBATCH -o logs/download-weatherbench-%j.out
#SBATCH --mail-user=salvaruehling@gmail.com
#SBATCH --mail-type=begin,end,fail

cd $CFS/m3504
mkdir -p weatherbench
cd weatherbench
wget 'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg&files=all_5.625deg.zip' -O all_5.625deg.zip --no-check-certificate
unzip all_5.625deg.zip