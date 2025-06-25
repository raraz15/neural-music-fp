source /usr/local/conda/etc/profile.d/conda.sh

# # Exit immediately if a command exits with a non-zero status.
# set -e

########################################################################

if [ $# == 0 ]; then
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
    exit 0
fi

########################### PARAMETERS #################################

# This script uses the 100th epoch by default. Change if needed.
EPOCH=100

########################### INPUTS #####################################

SYNTH_QUERY_DIR=../datasets/neural-music-fp-dataset/music/test/queries/clean-time_shifted-degraded/
SYNTH_DB_DIR=../datasets/neural-music-fp-dataset/music/test/database/

########################### OUTPUTS ####################################

LOG_ROOT_DIR=logs/nmfp/fma-nmfp_deg
EMB_DIR=$LOG_ROOT_DIR/emb

SYNTH_EMB_DIR=$EMB_DIR/nmfp/$4/$EPOCH/
SYNTH_EMB_QUERY_DIR=$SYNTH_EMB_DIR/queries/
SYNTH_EMB_DB_PATH=$SYNTH_EMB_DIR/database/database.mm
echo $SYNTH_EMB_QUERY_DIR
echo $SYNTH_EMB_DB_PATH

######################### ENVIRONMENT VARIABLES #########################

# Set GPU
export CUDA_VISIBLE_DEVICES=$1

conda activate nmfp

######################### COMMIT SHA ##################################

# Get current commit sha
commit_sha=$(git rev-parse HEAD)
echo $commit_sha

########################## WAIT ################################

# Sleep for $8 hours
if [ $5 != 0 ]; then

    echo "Sleeping for $5 hours..."
    sleep $(( $5 * 3600 ))

fi

########################### TRAINING ##################################
# Train

# If param3 is 1, train
if [ $3 == 1 ]; then

    # Stash any left changes
    echo "Stashing any left changes before training..."
    git stash save "Stash changes before running pipeline 'train'"

    # Checkout to original commit
    git checkout $commit_sha

    # Train
    echo "Training..."
    python train.py $2

fi

######################### GENERATION ###################################
# Generate Fingerprints on the synthethic test set

# Stash any left changes
echo "Stashing any left changes before generation..."
git stash save "Stash changes before running pipeline 'generate'"

# Checkout to original commit
git checkout $commit_sha

# Generate fingerprints
echo "Generating fingerprints..."
python evaluation-extraction.py $2 --queries $SYNTH_QUERY_DIR --database $SYNTH_DB_DIR --batch-size 2048

######################## MODEL EVALUATION ###############################
# Evaluate

# Stash any left changes
echo "Stashing any left changes before evaluation..."
git stash save "Stash changes before running pipeline 'evaluate'"

# Checkout to original commit
git checkout $commit_sha

# Evaluate
echo "Evaluating..."
python evaluation-retrieval.py $SYNTH_EMB_QUERY_DIR $SYNTH_EMB_DB_PATH

########################################################################

echo "Done!"
