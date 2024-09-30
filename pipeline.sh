source /usr/local/conda/etc/profile.d/conda.sh

conda activate tf

########################################################################

if [ $# == 0 ]; then
    echo "Description: Runs the full pipeline for a particular commit. 
        CUDA_VISIBLE_DEVICES=ID train.py CONFIG_PATH
        CUDA_VISIBLE_DEVICES=ID generate.py CONFIG_PATH ...
        CUDA_VISIBLE_DEVICES=ID evaluate.py ..."
    echo "Usage: $0 param1 param2 param3 param4 param5"
    echo "param1: GPU ID"
    echo "param2: Config path"
    echo "param3: Set to 1 for training a model and 0 for not training.
        If set to 0, the script will use the trained model in the config 
        path."
    echo "param4: Query audio dir"
    echo "param5: Database audio dir"
    echo "param6: Logs root dir. The logs will be saved in logs/fp/ and 
        logs/eval/ Expect the training logs, where this directory is 
        specified in the config file."
    echo "param7: Model name"
    echo "param8: Initial sleep time in hours"
    exit 0
fi

########################### DIRECTORIES #################################

# This script uses the 100th epoch by default. Change it if needed.
EPOCH=100
FP_DIR=$6/fp/$7/$EPOCH/
echo $FP_DIR

########################## GIT TRACKING ################################

# Get current commit sha
commit_sha=$(git rev-parse HEAD)
echo $commit_sha

# Sleep for $8 hours
if [ $8 != 0 ]; then

    echo "Sleeping for $8 hours..."
    sleep $(( $8 * 3600 ))

fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$1

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
# Generate Fingerprints

# Stash any left changes
echo "Stashing any left changes before generation..."
git stash save "Stash changes before running pipeline 'generate'"

# Checkout to original commit
git checkout $commit_sha

# Generate fingerprints
echo "Generating fingerprints..."
python generate.py $2 --query_chunks $4 --db_tracks $5

######################## MODEL EVALUATION ###############################
# Evaluate

# Stash any left changes
echo "Stashing any left changes before evaluation..."
git stash save "Stash changes before running pipeline 'evaluate'"

# Checkout to original commit
git checkout $commit_sha

# Evaluate
echo "Evaluating..."
python evaluate.py $FP_DIR

########################################################################

echo "Done!"
