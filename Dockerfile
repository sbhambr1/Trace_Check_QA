# Dockerfile for Trace-KD

# Start from the NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Define environments
ENV MAX_JOBS=16
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Define installation arguments
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini aria2 && \
    apt-get clean

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core grpcio

# Reinstall CUDA 12.4
RUN aria2c https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN aria2c --always-resume=true --max-tries=99999 https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 && \
    rm cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    update-alternatives --set cuda /usr/local/cuda-12.4 && \
    rm -rf /usr/local/cuda-12.6

RUN pip install --resume-retries 999 --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN pip install --resume-retries 999 --no-cache-dir "tensordict==0.6.2" torchdata "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff

# Install flash-attn-2.7.4.post1 (cxx11abi=False)
RUN wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fix packages
RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# Install cudnn
RUN aria2c --max-tries=9999 https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cudnn-cuda-12 && \
    rm cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb

# Install conda
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -tipsy

# Set conda path
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment
RUN conda create -n trace_kd python=3.10
RUN conda init bash && \
    echo "conda activate trace_kd" >> ~/.bashrc
RUN conda install -c conda-forge cudatoolkit-dev -y

