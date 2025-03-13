#!/bin/bash
#SBATCH -p RTX4060,RTX3080
#SBATCH -n 1
#SBATCH --gres=gpu:1 		

# Email notifications
#SBATCH --mail-user=s1934251@ed.ac.uk  # Your email address
#SBATCH --mail-type=END           # Types of notifications

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

module load cuda

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate openmm

srun python run_window.py $NUMBER