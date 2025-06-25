FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH="/opt/conda/bin:$PATH"

# Update system and install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget build-essential screen curl grep sed dpkg git tmux nano htop sysstat psmisc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

RUN mkdir -p \$CONDA_PREFIX/etc/conda/activate.d && \
echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/' > \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install conda packages
RUN conda create -n nmfp python=3.11 -y && \
    conda install -n nmfp -c conda-forge cudatoolkit=11.8 cudnn=8.9 faiss-gpu=1.7.4 pandas=2.1 -y && \
    conda install -n nmfp pyyaml -y && \
    conda clean --all -y

# Install pip packages
RUN conda run -n nmfp pip install --no-cache-dir tensorflow==2.13 soundfile essentia==2.1b6.dev1110

# The following lines are to shut up the annoying warnings from tensorflow
# Install cuda-nvcc
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nmfp && \
    conda install -c nvidia cuda-nvcc --yes"
# Find the correct libdevice path and copy
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate nmfp && \
    mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/ && \
    find $CONDA_PREFIX -name 'libdevice.10.bc' -exec cp -p {} $CONDA_PREFIX/lib/nvvm/libdevice/ \;"
# Set environment variable for XLA
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"

# Set up environment activation for interactive sessions
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate nmfp" >> ~/.bashrc

# For our code setup
RUN git config --global --add safe.directory /src/nafp_pp

CMD [ "/bin/bash" ]
