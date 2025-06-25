#!/bin/bash
#SBATCH -J NMFP-eval
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH --time=0-08:00:00
#SBATCH -o slurm/slurm_logs/%J.%N.out
#SBATCH -e slurm/slurm_logs/%J.%N.err

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the project conda environment
conda activate nmfp

######################################################################## Main ###############################################################################

QUERY_DIR=$1

DATABASE_DIR=$2

# Evaluate the generated fingerprints
python -u evaluation-retrieval.py $QUERY_DIR $DATABASE_DIR