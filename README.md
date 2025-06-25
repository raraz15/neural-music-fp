# Neural Music Fingerprinting

This repository contains the code to run the experiments detailed in our ISMIR2025 paper 'Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification'. We build upon the [Neural Audio Fingerprinting](https://github.com/mimbres/neural-audio-fp) framework, introducing several key improvements to create a more robust and efficient music identification system.

**Key Features**
- **Enhanced Robustness**: Models are trained to be resilient against various real-world audio degradations.
- **Modernized Stack**: The codebase is updated to use Python 3.11, TensorFlow 2.13, and Faiss 1.7.4.
- **Advanced Database Management**: Our implementation includes powerful FAISS utilities, allowing you to easily resume failed extractions and perform fast updates to fingerprint databases.
- **Reproducible & Easy to Deploy**: A pre-configured Docker image with GPU support is provided for hassle-free setup and deployment.
- **Optimized Performance**: We've implemented automatic mixed-precision training and inference, significantly speeding up the process. A *full training run completes in less than a day* on a 24GB GPU.
- **Pre-trained Models**: Get started immediately with our best-performing models, available on Zenodo.

**Quick Facts & Limitations**
- We provide a Docker Image with GPU support!
- We only support 8000 Hz audio for training and evaluation. Inference can be done for various sampling rates by the integrated resampling (Upsampling is not recommended).
- Training and evaluation is only possible with `.wav` files. For inference, we support `.wav`, `.flac`, `.mp3`, `.aac`, and `.ogg` audio formats. But the model was tested with `.mp3` and `.wav` files only.
- We **highly recommend** checking our database management code for FAISS. It allows various functionalities that are not implemented in other fingerprinting repositories, e.g., continue a failed database extraction process, fast updating a database with new tracks.
- We provide bash scripts that perform the entire train + evaluation pipeline in `pipeline.sh`.
- We also provide slurm scripts in `slurm/`
- You may encounter hanging process issues after fingerprint extraction due to tensorflow and multiprocessing, we tried taking care of it but feel free to open an issue.
- Training and inference could be made much faster on the GPU. When I started the project I only had access to GPUs with 12GB memory so I had to keep the entire input extraction pipeline on the CPU. Still, I optimized the processing as much as I could. In addition, I implemented automatic mixed precision training and inference, which speeds up the training considerably. Overall, with a 24GB memory GPU, training takes less than a day.

## Table of Contents

- [Installation](#installation)
  - [Conda Environment](#conda-environment)
    - [GPU support](#gpu-support)
      - [Possible Errors](#possible-errors)
    - [CPU only](#cpu-only)
  - [Docker Image](#docker-image)
- [Pre-trained Models](#pre-trained-models)
- [Usage](#usage)
  - [Inference with a Pre-trained Model](#inference-with-a-pre-trained-model)
    - [Extract Fingerprints](#extract-fingerprints)
    - [Perform Retrieval](#perform-retrieval)
  - [Reproducing the Paper's Experiments](#reproducing-the-papers-experiments)
    - [Train a Neural Audio Fingerprinter](#train-a-neural-audio-fingerprinter)
    - [Evaluation](#evaluation)
      - [Extraction](#extraction)
      - [Retrieval and Metrics](#retrieval-and-metrics)
- [Dataset for Training and Evaluation](#dataset-for-training-and-evaluation)
- [Cite](#cite)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Conda Environment

For the [Docker image](#installation-docker-image) please see.

We provide an `environment.yaml` file which can be used directly, but installing `faiss` and `tensorflow` together was challenging with the other packages. Therefore, I recommend following the indicated steps below.

**NOTE:** `pandas` and `matplotlib` are not crucial for most of the code. So you could just comment out some lines and be ready.

#### GPU support

1. Create the environment

    ```bash
    conda create -n afp python=3.11
    conda activate afp
    ```

1. Install cuda dependencies for tensorflow 2.13

    ```bash
    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

    You need to log off to close the session and log in again.

1. Install faiss and other dependencies

    ```bash
    conda activate afp
    conda install -c conda-forge faiss-gpu=1.7.2
    conda install pyyaml scipy=1.11.4 pandas=2.1.4 matplotlib=3.8
    ```

1. Install tensorflow

    ```bash
    pip install tensorflow==2.13
    #Verify install:
    python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
    ```

1. Install essentia

    ```bash
    pip install essentia==2.1b6.dev1110
    ```

##### Possible Errors

**Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice**

If you get this [error](https://github.com/tensorflow/tensorflow/issues/58681) you need to additionaly follow these steps.

```bash
# The fix for training models
conda install -c nvidia cuda-nvcc --yes
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
```

#### CPU only

```bash
# Create the environment
conda create -n afp python=3.11
conda activate afp
```

```bash
# Add the LD_LIBRARY_PATH to the path
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# You need to log off to close the session and log in again at this point
```

```bash
# Install the packages
conda activate afp
conda install -c conda-forge faiss-cpu=1.7.2
conda install pyyaml scipy=1.11.4 pandas=2.1.4 matplotlib=3.8
pip install tensorflow-cpu==2.13
pip install essentia==2.1b6.dev1110
```

### Docker Image

This project comes with a pre-configured `Dockerfile` and `docker-compose.yml` for reproducible experiments. You will need to run the following lines from the project directory.

1. Adjust docker-compose.yml

    Update the following volume paths in docker-compose.yml to match your local filesystem:

    ```docker
    volumes:
      - /path/to/your/data:/src/datasets         # Where your datasets are
      - /path/to/output/logs:/src/nmfp/logs      # Where outputs/logs will be saved
      - /path/to/this/repo:/src/nmfp             # Mount this repository
    ```

    You can also leave default paths like ./datasets and ./logs if running locally with a standard folder layout.

1. Build the Docker image

    Run: `make build`

    This will:
    - Clean any existing containers with the same name
    - Rebuild the image using the `Dockerfile`
    - Set up all dependencies and environments

1. Start the container

    `docker compose up -d`

1. Access the container

    `docker exec -it nmfp bash -c 'cd /src/nafp_pp && exec bash'`

## Pre-trained Models

Our pre-trained models (~200MB each after decompression) are available at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15719945.svg)](https://doi.org/10.5281/zenodo.15719945)

You can use use `./download-models.sh` to automatize the process. Make the script executable and run it:

```bash
chmod +x download-models.sh
./download-models.sh
```

The best performing model is `nmfp-triplet`, we also included `nmfp-multiposcon` for comparison.

## Usage

The project supports two main workflows: direct inference for fingerprinting and retrieval, and the full experimental pipeline to reproduce our paper's results.

**NOTE** on potential issues: Due to interactions between TensorFlow and Python's multiprocessing, you may encounter hanging processes after fingerprint extraction. We have implemented fixes, but if you experience this, please feel free to open an issue.

### Inference with a Pre-trained Model

This is the fastest way to use the fingerprinter on your own audio files. It involves two steps: extraction and retrieval.

#### Extract Fingerprints

Use `extraction.py` to generate fingerprints from your audio files (supports single files, directories, or lists of paths).

Basic usage:
```bash
python extraction.py \
  audio_path_or_dir \
  logs/nmfp/fma-nmfp_deg/checkpoint/nmfp-triplet/config.yaml \
  /path/to/output/dir \
```

Advanced usage:
```bash
# Advanced: Distribute extraction across multiple processes/machines
# For example, to run the 2nd of 4 jobs:
python extraction.py \
  <path/to/your/audio_file_or_dir> \
  pretrained_models/nmfp-triplet/config.yaml \
  <path/to/output/embeddings_dir> \
  --num-partitions 4 \
  --partition 2
```

Refer to `python extraction.py -h` for more details.

#### Perform Retrieval

Search for query fingerprints against a database index using `retrieval.py`. It will:
- load the fingerprints for a single query audio clip or multiple query audio clips
- query *all* the fingerprint segments in each `.npy` file against the database and return the top-k approximately most similar fingerprints for each one independently. 

To build the FAISS index, you must provide the database in one of three ways:
- `--database-embeddings`: A directory of `.npy` fingerprint files.
- `--database-memmap`: provide a path to the `database.mm` memmap file where all the fingerprints of individual audio clips are merged together.
- `--database-index` A pre-built (trained and populated) `.index` FAISS file.

Usage:
```bash
python retrieval.py /path/to/query_file_or_directory output_dir --database-index database.index
```

The output will be saved in `output_dir/results.csv` with the following fields:
```csv
query_fp_path,query_seq_len,pred_track_path,pred_start_segment,score
```

Refer to `python retrieval.py -h` for more details.

### Reproducing the Paper's Experiments

In `pipeline.sh`, we put together the training and evaluation pipeline, it can be used as an example for the following.

#### Train a Neural Audio Fingerprinter

We use `.yaml` config files to define the parameters related to the architecture, the model, data, and training. Below is the usage for training a model. This will save the training logs and model checkpoints to the directory specified in the config with `cfg['MODEL']['LOG_ROOT_DIR']`. The trained model's configuration will be saved next to the weights.

Usage:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/nmfp-triplet.yaml
```

#### Evaluation

Evaluation is performed in two steps: fingerprint extraction and retrieval.

##### Extraction

Use `evaluation-extraction.py` to generate fingerprints for the database and query sets. This script efficiently creates a single `database.mm` memmap file for all database tracks and individual `.npy` files for queries.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation-extraction.py logs/nmfp/fma-nmfp_deg/checkpoint/nmfp-triplet/config.yaml \
--queries neural-music-fp-dataset/music/test/queries/clean-time_shifted-degraded/ \
--database neural-music-fp-dataset/music/test/database/
```

##### Retrieval and Metrics

Use `evaluation-retrieval.py` to evaluate the retrieval performance. The script computes the following metrics:

* track-level `track_hit_rate.txt`
    * Top-1 Hit rate
* segment-level `segment_hit_rate.txt`
    * Exact Top-1 Hit rate (query and reference are exactly time aligned)
    * Near Top-1 Hit rate (query and reference are misaligned less than 500 ms)
    * Far Top-1 Hit rate (query and reference are misaligned more than 500 ms, same as track-level)

This script can be used with arbitrary datasets (such as the industrial dataset mentioned in the paper). In this case you might choose to disable the segment level evaluation do to lack of time stamps. Nonetheless, the script can automatically disable this feature too.

The script will also create `analysis.csv`. You can use this to listen to the matches or do other types of analyses.

Header of `analysis.csv`
```csv
query_audio_path,query_start_segment,seq_start_idx,query_chunk_bound,seq_len,gt_track_path,pred_track_path,pred_start_segment,score
```

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation-retrieval.py logs/nmfp/fma-nmfp_deg/emb/nmfp-triplet/100/queries/ logs/nmfp/fma-nmfp_deg/emb/nmfp-triplet/100/database/
```

## Dataset for Training and Evaluation

For music, we use the [FMA dataset](https://github.com/mdeff/fma), for audio degradation we use multiple datasets. For more information, please check `dataset_creation/README.md`.

You can use use `./download-dataset.sh` to download the `.tar.gz` file (48GB compressed) and decompress it. Make the script executable and run it:

```bash
chmod +x download-dataset.sh
./download-dataset.sh
```

We share the following data in zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15736620.svg)](https://doi.org/10.5281/zenodo.15736620):
* music
  * train chunks (10,000 x 30 sec)
  * test queries (10,000 x 30 sec) (clean-time_shifted-degraded)
    * The time boundary of the chunk inside the full track, which you can use to get the clean versions. We couldn't share the steps in between due to zenodo's 50GB cap.
  * Full tracks of the queries
* degradation

However, the entire test database files take 400+ GB space. You should download the FMA dataset and process them by following the steps in `dataset_creation/README`. We included the full tracks of the query chunks so that the clean versions are exactly the same (During mp3 to wav conversion and processing sox may apply dithering, which is a stochastic process. Not sure about the effect of this, but ideally, master tracks should be the same.)

**NOTE**: We do not perform validation during training therefore we did not include NAFP's validation partition with the dataset.

## Cite

Please cite the following publication when using the code, data, or the models:

> R. O. Araz, G. Cortès-Sebastià, E. Molina, J. Serrà, X. Serra, Y. Mitsufuji, and D. Bogdanov, “Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification,” in Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), 2025.

```bibtex
@inproceedings{araz_enhancing_2025,
  title     = {Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification},
  author    = {Araz, R. Oguz and Cortès-Sebastià, Guillem and Molina, Emilio and Serrà, Joan and Serra, Xavier and Mitsufuji, Yuki and Bogdanov, Dmitry},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2025}
}
```

## License

* The code in this repository is licensed under the [Affero GPLv3](https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0) license.
* The model weights are also licensed under the [Affero GPLv3](https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0) license.

See the LICENSE file for details.

## Acknowledgements

This work was supported by the pre-doctoral program AGAUR-FI ajuts (2024 FI-3 00065) Joan Oró, funded by the Secretaria d’Universitats i Recerca of the Departament de Recerca i Universitats of the Generalitat de Catalunya; and by the Cátedras ENIA program “IA y Música: Cátedra en Inteligencia Artificial y Música” (TSI-100929-2023-1), funded by the Secretaría de Estado de Digitalización e Inteligencia Artificial and the European Union – Next Generation EU.

This work was also part of the project TROBA – Technologies for the recognition of musical works in the era of dynamic generation of audio content (ACE014/20/000051), within the call Nuclis d’R+D 2024, with the support of ACCIÓ (Agency for Business Competitiveness, Government of Catalonia).
