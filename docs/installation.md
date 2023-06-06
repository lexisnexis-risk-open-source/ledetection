# Installation Guide
In this guide we show how to prepare and build the LEDetection project, which has been tested on Ubuntu>=18 with the following dependencies: Python>=3.7, PyTorch>=1.6.0, and CUDA>=10.1 with cuDNN>=7.

As a prerequisite, we first clone the LEDetection repository to our local home directory:

```bash
git clone \
    https://github.com/lexisnexis-risk-open-source/ledetection.git \
    ~/ledetection
# Since we develop and run ledetection directly,
# we create some new directories inside ledetection
# to store development artifacts.
cd ~/ledetection
mkdir results work_dirs
```

## Best Practices with Docker
We recommend using Docker to containerize all the complex dependencies when building this project. We provide example [Dockerfiles](https://github.com/lexisnexis-risk-open-source/ledetection/tree/main/docker) pertaining to whether CUDA 10.x or CUDA 11.x is used.

- For Ampere GPU architectures (compute capability 8.6), such as GeForce 30 series and NVIDIA A100, CUDA 11.x is required.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.x offers better compatibility and is more lightweight.

Make sure the GPU driver satisfies the minimum version requirements, according to [these NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Also, ensure that the [docker version](https://docs.docker.com/engine/install/) is >=19.03.

### Build

```bash
# The default is to build an image with Python 3.8,
# PyTorch 1.11, CUDA 11.3, and cuDNN 8.
docker build \
    --build-arg USER_ID=$(id -u) \
    -t ledetection:pytorch1.11.0-cuda11.3 \
    docker/CUDA11/
```

```bash
# Alternatively, we can build with Python 3.7,
# PyTorch 1.6, CUDA 10.1, and cuDNN 7.
docker build \
    --build-arg USER_ID=$(id -u) \
    -t ledetection:pytorch1.6.0-cuda10.1 \
    docker/CUDA10/
```

### Usage
We recommend running Docker as a user mapped from our local machine to the container via the argument `-u $(id -u)`, where the `bash` command `id -u` gives the user ID on the local host. Below is an example `docker run` command to execute a LEDetection training job on the VOC dataset using 2 GPUs.

```bash
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0,1"' \
    -u $(id -u) --rm --ipc=host \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    bash tools/dist_train_ssod.sh \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py \
    2 FasterRCNN NA NA
```

Here, we assume that the VOC data source (and others) is stored on the local machine at the path `/data/shared/`. We use the `docker run -v` flag to map volumes between the local host and the container at runtime. Our recommended best practice is to map two volumes from the local host to be used by the container:

1. We map the entire local LEDetection repository to the container so that any local modifications will take effect inside the container and can be used by the container at runtime.

    ```bash
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection
    ```

2. We map *data volumes* to a specified location inside the container so it can access data not previously copied during runtime.

    ```bash
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data
    ```

Then, in our config files, we just need to point to the proper paths of data sources and other artifacts needed by the job, which are *relative to the working directory of the container*. The default working directory of the container is set by `docker run -w ${APP_HOME_DIR}/ledetection`. If everything is configured correctly, we should be able to train and test models using LEDetection!

## Anaconda Environment
Alternatively, we can install LEDetection and its dependencies using an Anaconda environment. For the sake of clarity, we assume all installation steps are conducted from the directory `/workspace`, although you can complete the installation from any directory you wish.

**Step 1.** Download and install Anaconda from the [official website](https://www.anaconda.com/products/distribution).

**Step 2.** Clone the `ledetection` repository into `/workspace`.

```bash
cd /workspace \
    && git clone https://github.com/lexisnexis-risk-open-source/ledetection.git
# Since we develop and run ledetection directly,
# we create some new directories inside ledetection
# to store development artifacts.
cd ledetection \
    && mkdir results work_dirs
```

**Step 3.** Clone `mmdetection` into `/workspace` to enable access to the `mmdetection` configuration files, which are needed for training models. We support `mmdetection>=2.16.0,<=2.28.0`.

```bash
cd /workspace \
    && git clone https://github.com/open-mmlab/mmdetection.git \
    && cd mmdetection \
    && git checkout v2.28.0 \
    && cd ..
```

**Step 4.** Create the conda environment.

```bash
cd /workspace/ledetection
conda env create -f environment-cpu.yaml
```

or

```bash
cd /workspace/ledetection
conda env create -f environment-gpu.yaml
```

**Step 5.** Verify the installation.

```bash
cd /workspace/ledetection

# No import error.
python -c "import ledet; print(ledet.__version__)"
# Example output: 0.0.1
python -c "import mmdet; print(mmdet.__version__)"
# Example output: 2.28.0
```