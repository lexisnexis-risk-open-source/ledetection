ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0+PTX;7.5;8.0;8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
# The default location for Torch Hub models is ~/.cache/torch/
# We can specify a different location by uncommenting the line below.
# ENV TORCH_HOME="data/torch/"

# Update NVIDIA signing keys
RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Update and install OS packages
RUN apt-get update && apt-get install -y ffmpeg wget vim git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root appuser
# Obtain your USER_ID from bash command `id -u`
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
WORKDIR /home/appuser

# Install MMCV
RUN pip install --no-cache-dir --user --upgrade pip wheel setuptools
RUN pip install --no-cache-dir --user mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install dependencies and useful packages
RUN conda clean -y --all
RUN pip install --no-cache-dir --user git+https://github.com/lvis-dataset/lvis-api.git
RUN pip install --no-cache-dir --user seaborn jupyter tensorboard scikit-video scikit-learn pandas lmdb nltk natsort shapely

# Install mmdetection and pin to v2.16.0
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection
WORKDIR /home/appuser/mmdetection
RUN git checkout v2.16.0
RUN pip install --no-cache-dir --user -r requirements/build.txt
RUN pip install --no-cache-dir --user -r requirements/optional.txt
RUN pip install --no-cache-dir --user -e .

WORKDIR /home/appuser

# Install ledetection
RUN git clone https://github.com/lexisnexis-risk-open-source/ledetection.git ledetection
WORKDIR /home/appuser/ledetection
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user -e .

