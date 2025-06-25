#!/bin/bash
#SBATCH -J NMFP-train
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --time=2-00:00:00
#SBATCH -o slurm/slurm_logs/%J.%N.out
#SBATCH -e slurm/slurm_logs/%J.%N.err

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the project conda environment
conda activate nmfp

######################################################################## Arguments ####################################################################

CONFIG=$1

# If a Git SHA ($2) is provided, checkout that commit
if [ -n "$2" ]; then
    SHA=$2
    echo "Git SHA provided: $SHA"
    git stash
    git checkout $SHA
    echo
fi

######################################################################## Main ###############################################################################

# Train a model
python -u train.py $CONFIG --workers 6 --queue 24
