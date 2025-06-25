#!/bin/bash
#SBATCH -J NMFP-gen
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=72g
#SBATCH --time=0-08:00:00
#SBATCH -o slurm/slurm_logs/%J.%N.out
#SBATCH -e slurm/slurm_logs/%J.%N.err

######################################################################## Anaconda ####################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the project conda environment
conda activate nmfp

######################################################################## Arguments ####################################################################

CONFIG=$1

QUERY_DIR=$2

DATABASE_DIR=$3

OUTPUT_ROOT=$4

# Check if a second input was provided
if [ $# -ge 5 ]; then
  SHA=$5
  echo "Git SHA provided: $SHA"
  git stash
  git checkout $SHA
  echo
fi

######################################################################## Generate ####################################################################

# Generate fingerprints
python -u evaluation-extraction.py $CONFIG \
  --queries $QUERY_DIR \
  --database $DATABASE_DIR \
  --output-root-dir $OUTPUT_ROOT\
  --workers 6 \
  --queue 24 \
  --batch-size 2048 \
