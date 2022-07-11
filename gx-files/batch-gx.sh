#!/bin/bash
#SBATCH --job-name=GX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
##SBATCH --time=00:50:00
#SBATCH --time=02:00:00
##SBATCH -o pipeline.slurm
#SBATCH -o ./gxJob.out.%j
#SBATCH -e ./gxJob.err.%j
GK_SYSTEM=traverse
export GK_SYSTEM

module purge
module load cudatoolkit/11.3
module load openmpi/gcc/4.0.4/64
module load netcdf/gcc
module load hdf5/gcc/openmpi-4.0.4/1.10.6
module load gsl/1.15
module load anaconda3/2021.5

#tag=gonzalez-2021_alpha0.0
#srun ./gx $tag > log.$tag
