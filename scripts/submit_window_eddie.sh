#!/bin/bash

# Max runtime 
#$ -l h_rt=10:00:00

# Set working directory to the directory where the job is submitted from:
#$ -cwd
#$ -N Umbrella_sampling
#$ -M s1934251@sms.ed.ac.uk

# Request one GPU:
#$ -q gpu
#$ -l gpu=1

#$ -l h_vmem=24G

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda

# Initialize a variable to store the number
NUMBER=0

# Use flag r to store the specified r0 value
while getopts "r:" opt; do
  case $opt in
    r)
      NUMBER=$OPTARG
      ;;
    \?)
      echo "Usage: $0 -r <number>"
      exit 1
      ;;
  esac
done

# Shift off the processed options
shift $((OPTIND -1))

# Activate the OpenMM conda environment
source /exports/csce/eddie/chem/groups/Michel/anaconda/envs/openmm_coils/bin/activate
conda activate /exports/csce/eddie/chem/groups/Michel/anaconda/envs/openmm_coils

# Run python
/exports/csce/eddie/chem/groups/Michel/anaconda/envs/openmm_coils/bin/python run_window.py $NUMBER
