# Dataset Creation

This section explains how the dataset used for our experiments was created. We leverage the FMA dataset for music tracks and follow NAFP’s partitioning closely for train and validation splits, while our query subset for the test set is an expanded version of NAFP’s. Therefore, reproducing this dataset requires both the FMA and NAFP datasets. Additionally, various audio degradation datasets are employed, which are detailed in subsequent sections.

## Conversion Improvements

For converting MP3 files to WAV, we initially used the bash script provided by NAFP. However, due to the lack of automatic clip guarding during MP3 decoding, many files experienced clipping. To address this, we implemented our own version with clip prevention. We created separate scripts for music files and Impulse Responses (IR). Specifically, stereo IR files are split into left (L) and right (R) channels and saved as individual mono files. This avoids phase cancellation and captures distinct room responses from each channel.

## Music Audio

All the music files in this project were taken from the [FMA dataset](https://github.com/mdeff/fma). We adopted the development and evaluation splits as proposed in NAFP but with the following minor chages:

- We avoided clipping distortion that occurs during wav conversion as opposed to NAFP
- We removed the bitwise exact duplicates (5 files in the training set, none in validation)

Steps to reproduce:

1. Download FMA dataset's full partition.
1. Download FMA dataset's genre and track annotations.
1. Get a hash dictionary of the bit-wise duplicates in FMA dataset using `exact_duplicate_finder.py`.
1. Convert all the tracks from `mp3` to `wav` format using the `convert_to_8k_wav_mono.sh` script.
1. Download NAFP's mini dataset which contains the development tracks, 500 test query tracks and some database tracks. We will not use these files themeselves, but will use the filenames to ensure consistency.

### Training Set

We used the same training tracks as NAFP, and (probably) the same chunks, NAFP states that they used FMA_small and fma_medium for the development tracks, and these tracks are already about 30s audio chunks sampled from `fma_full`. However, since their conversion results in clipping and their splits contain bitwise exact duplicates, we follow these steps.

Steps to reproduce:

1. Download *fma_medium*
1. Convert fma_medium files to wav format with `dataset_creation/convert_to_8k_wav_mono.sh`.
1. Remove the duplicates from the NAFP train and val set using `remove_dev_duplicates.py`
1. Get the filenames of the remaining train and val set audio.
1. Copy the corresponding files from `fma_medium-wav_8khz_16bit/` to a new directory.

This way we made sure that the tracks do not clip during conversion, our training partition do not contain duplicates, and the audio chunks are time aligned with the NAFP partitions.

### Evaluation Set

The test set consists of two parts: a database of 95,163 full tracks and 30-second audio chunks sampled from a subset of 10,000 database tracks for querying. The query set contains the 500 query tracks used originally in NAFP and we additionally add 9,500 random tracks from the database.

Steps to reproduce:

1. Create `test_queries.csv` and `test_database.csv` files using `extend_nafp_query_set.py`
1. Run `create_degraded_query_set.py` to sample chunks from the query tracks and degrade them. (For downloading the degradation files and processing them, read the next [section](#audio-degradation).)

Since in our experiments, we compare the effect of NAFP's degradations vs our degradations, this script is used twice, once for each degradation. The nice part is that this script ensures that the sampled chunks are aligned in time between different degradations.

## Audio Degradation

This section contains the information related to the audio degradation datasets. For a detailed analysis of NAFP's and our degradations' use this [notebook](./dataset_creation/degradation_analysis_and_split.ipynb). It also contains the code to split files into train, validation, and test sets.

*NOTE:* The notebook still has some cleaning to do. Soon we will host the files on zenodo, and share the file names here for increased reproducibility.

### Background Noise

1. Download TUT2016 [development](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/tut_asc_2016_eval.html) and [evaluation sets](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/tut_asc_2016_dev.html)
1. Convert all the mp3 files to wav files using `convert_to_8k_wav_mono.sh`

We use the provided development and evaluation splits by the original authors.

### Room Impulse Responses

We used the following 3 datasets. From each dataset we selected certain files and then converted them to mono 8kHz. We explain the motivation behind these selections below. You can use [https://github.com/RoyJames/room-impulse-responses](https://github.com/RoyJames/room-impulse-responses) to automate downloading the files.

### MIT

[https://mcdermottlab.mit.edu/Reverb/IR_Survey.html](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)

1. Converted all the recordings with `convert_to_8k_wav_mono.sh` since they are all mono.
1. We split the recordings based on their durations.

### OpenAIR

[https://www.openair.hosted.york.ac.uk/](https://www.openair.hosted.york.ac.uk/)

1. We started by choosing the mono and stereo recordings.
1. Converted the mono files with `convert_to_8k_wav_mono.sh`
1. Converted the stereo files with `convert_stereo_RIR_to_8k_wav_mono.sh`.
1. We split the rooms into train and test sets based on their durations.

### Aachen IR (AIR)

[https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)

1. We created audio files from the `.mat` files using [https://github.com/jonashaag/RealRIRs](https://github.com/jonashaag/RealRIRs)
1. We excluded the measurements done with a dummy head.
1. We included both channels from binaural recordings separately as mono.
1. Converted the selected recordings with `convert_to_8k_wav_mono.sh`.
1. We put the recordings from 'meeting', 'lecture', 'stairway', and 'office' to the train set while 'aula_carolina' and 'booth' were placed in the test set.

### Microphone Impulse Responses

[https://zenodo.org/records/4633508](https://zenodo.org/records/4633508)

1. We only use the normalized recordings.
1. We only keep IR recordings at incident angles at multiple of 60 degrees.
1. Recordings are already mono, so we use `convert_to_8k_wav_mono.sh` for conversion.
1. We split the microphones.
