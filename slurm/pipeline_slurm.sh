#!/bin/bash
#SBATCH -J NMFP-pipe
#SBATCH -p impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=96g
#SBATCH --time=2-00:00:00
#SBATCH -o slurm/slurm_logs/%J.%N.out
#SBATCH -e slurm/slurm_logs/%J.%N.err

if [ $# -lt 5 ]; then
    echo "Description: Runs the full pipeline for a particular commit. 
        CUDA_VISIBLE_DEVICES=ID train.py CONFIG_PATH
        CUDA_VISIBLE_DEVICES=ID evaluation-extraction.py CONFIG_PATH ...
        CUDA_VISIBLE_DEVICES=ID evaluation-retrieval.py ..."
    echo "Usage: $0 param1 param2 param3 param4 param5"
    echo "param1: GPU ID"
    echo "param2: Config path"
    echo "param3: Set to 1 for training a model and 0 for not training.
        If set to 0, the script will use the trained model in the config 
        path."
    echo "param4: Model name"
    echo "param5: Initial sleep time in hours"
    echo "param6: Query audio directory for generation"
    echo "param7: Database audio directory for generation"
    exit 1
fi

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

conda activate nmfp

########################### INPUTS #####################################

QUERY_DIR=$6
DB_DIR=$7

########################### PARAMETERS #################################

# This script uses the 100th epoch by default. Change if needed.
EPOCH=100

########################### OUTPUTS ####################################

LOG_ROOT_DIR=logs/nmfp/fma-nmfp_deg
EMB_DIR=$LOG_ROOT_DIR/emb

SYNTH_EMB_DIR=$EMB_DIR/nmfp/$4/$EPOCH/
SYNTH_EMB_QUERY_DIR=$SYNTH_EMB_DIR/queries/
SYNTH_EMB_DB_PATH=$SYNTH_EMB_DIR/database/database.mm
echo $SYNTH_EMB_QUERY_DIR
echo $SYNTH_EMB_DB_PATH

###################### Main ###########################################

# Get the current commit SHA
SHA=$(git rev-parse HEAD)

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=0

########################## WAIT ################################

# Sleep if the 5th argument is non-zero
if [ "$5" -ne 0 ]; then
  echo "Sleeping for $5 hours…"
  sleep $(( $5 * 3600 ))
fi

########################### TRAINING ##################################
# Train

# If param3 is 1, train
if [ "$3" -eq 1 ]; then

    # Stash any left changes
    echo "Stashing any left changes before training..."
    git stash save "Stash changes before running pipeline 'train'"

    # Checkout to original commit
    git checkout $SHA

    # Train
    echo "Training..."
    python -u train.py $2 -w 6 -q 24

fi

######################### GENERATION ###################################
# Generate Fingerprints on the synthethic test set

# Stash any left changes
echo "Stashing any left changes before generation..."
git stash save "Stash changes before running pipeline 'generate'"

# Checkout to original commit
git checkout $SHA

# Generate fingerprints
echo "Generating fingerprints..."
python -u evaluation-extraction.py $2 --queries $QUERY_DIR --database $DB_DIR --batch-size 2048 -w 6 -q 24

######################## MODEL EVALUATION ###############################
# Evaluate

# Stash any left changes
echo "Stashing any left changes before evaluation..."
git stash save "Stash changes before running pipeline 'evaluate'"

# Checkout to original commit
git checkout $SHA

# Evaluate
echo "Evaluating..."
python -u evaluation-retrieval.py $SYNTH_EMB_QUERY_DIR $SYNTH_EMB_DB_PATH

########################################################################

echo "Done!"
