#!/bin/bash
#SBATCH --nodes=3                                                                  
#SBATCH --ntasks-per-node=4                                                        
#SBATCH --gpus-per-task=1
#SBATCH -C gpu
#SBATCH -A ntrain1_g
#SBATCH -J TrinityGX                                                                                                                
#SBATCH --time=02:00:00
#SBATCH --qos=regular                                                              
                                                                                   
export GK_SYSTEM=perlmutter
module purge
module load PrgEnv-gnu/8.3.3                                                       
module load cudatoolkit/11.5
module load nvhpc-mixed/21.11                                                      
module load cray-hdf5-parallel/1.12.1.1                                            
module load cray-netcdf-hdf5parallel/4.8.1.1                                       
module load e4s/21.11
module load gsl/2.7                                                                
module load python/3.9-anaconda-2021.11                                            
module load craype-x86-milan                                                       
module load craype-accel-nvidia80                                                  
module load cray-fftw/3.3.8.13    


tag=jet-example
python run.py $tag.in > log.$tag
