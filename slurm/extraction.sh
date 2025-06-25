#!/bin/bash
#SBATCH -J NMFP-gen
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=72g
#SBATCH --time=0-12:00:00
#SBATCH -o slurm/slurm_logs/%A.%N.out
#SBATCH -e slurm/slurm_logs/%A.%N.err

######################################################################## Anaconda ####################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the project conda environment
conda activate nmfp

######################################################################## Arguments ####################################################################

AUDIO_DIR=$1
CONFIG=$2
OUTPUT_DIR=$3
PARTITIONS=$4
IDX=$5

######################################################################## Generate ####################################################################
# For some reason my SLURM could not work with a job array so I use a single job with a partition argument.


# Generate fingerprints
python -u extraction.py $AUDIO_DIR $CONFIG $OUTPUT_DIR \
  --num-partitions $PARTITIONS \
  --partition $IDX \
  --workers 6 \
  --queue 24 \
  --batch-size 2048 \
