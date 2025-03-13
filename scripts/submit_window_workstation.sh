#!/bin/bash
#SBATCH -p main
#SBATCH -n 1
#SBATCH --gres=gpu:1 		
#SBATCH --mem=4G   

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

source /home/benedicttan/miniforge3/etc/profile.d/conda.sh
conda activate openbiosim

python run_window.py $NUMBER