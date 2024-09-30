#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=187G
#SBATCH --time=00-24:00            # time (DD-HH:MM)
#SBATCH --output=/home/kmcguiga/projects/def-sirisha/kmcguiga/computeCanadaOutput/%j.out
#SBATCH --account=def-sirisha
#SBATCH --mail-user=kmcguiga@uwaterloo.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
module purge
module load StdEnv/2023
module load gcc/12.3
module load eccodes/2.31.0
module load openmpi/4.1.5
module load mpi4py
module load hdf5/1.14.2
module load netcdf/4.9.2
source /home/kmcguiga/projects/def-sirisha/kmcguiga/environments/gsif/bin/activate
python experiment.py