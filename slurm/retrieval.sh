#!/bin/bash
#SBATCH -J NMFP-gen
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH --time=0-04:00:00
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

QUERIES=$1
OUTPUT_DIR=$2
shift 2  # shift out the first two args so "$@" only contains the extras
# you can pass "--database-embeddings db_emb" or "--database-index db_idx", ...

######################################################################## Retrieval ####################################################################

python -u retrieval.py "$QUERIES" "$OUTPUT_DIR" "$@"