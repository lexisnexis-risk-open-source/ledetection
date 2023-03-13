# Quickstart
In this guide we provide an introduction on how to:

* Set up a basic workspace with LEDetection;
* Train a supervised and semi-supervised detector using pre-built configurations;
* Evaluate the trained models.

Before we begin, we recommend the reader study the documentation and tutorials on [MMDetection](https://mmdetection.readthedocs.io/en/stable/) since this project mirrors its usage design. As such, we do assume some familiarity with the basic functionality of MMDetection.

## Workspace Setup
We recommend configuring a workspace to store data, logs, model artifacts, etc. according to the outline below. We suggest storing the datasets, like COCO and VOC, somewhere outside the project directory and `symlink` the dataset root to `${LEDETECTION_HOME}/data`. If your project directory structure is different, you may need to change the corresponding paths in the config files.

```
ledetection/                # The project working directory.
├── configs/                # Where configurations live.
├── ledet/                  # The source code.
├── results/                # Store results and artifacts here.
├── tools/                  # Utilities for train, test, and other functions.
├── work_dirs/              # Where model checkpoints and text logs will be saved.
├── data/                   # Datasets live here. Symlink data directories as necessary.
│   ├── coco/               # The COCO data structure.
│   │   ├── annotations/
│   │   ├── train2017/
│   │   ├── val2017/
│   │   ├── unlabeled2017/
│   ├── VOCdevkit           # The PASCAL VOC data structure.
│   │   ├── VOC2007/
│   │   ├── VOC2012/
```

## Training and Evaluation
Following MMDetection usage, we train and evaluate detection models via configuration. LEDetection supports pre-built configurations for supervised, semi-supervised, and few-shot detection. In addition, all base MMDetection configurations can be run within LEDetection. Essentially, LEDetection is a wrapper for MMDetection, and can do everything MMDetection does, but further augments its portfolio with select semi-supervised and few-shot detection capabilities.

### Sample Pre-Built Configurations
***Note: All settings are configured for 8x multi-GPU training.***

| Config Name | Labeled Dataset | Unlabeled Dataset | Detection Mode |
|-------------|-----------------|-------------------|--------|
| [faster\_rcnn\_r50\_caffe\_fpn\_voc07.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py) | VOC07 | N/A | Supervised |
| [faster\_rcnn\_r50\_caffe\_fpn\_voc0712.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/supervised/faster_rcnn_r50_caffe_fpn_voc0712.py) | VOC0712 | N/A | Supervised |
| [semi\_supervised\_faster\_rcnn\_r50\_caffe\_fpn\_voc.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/semi_supervised/semi_supervised_faster_rcnn_r50_caffe_fpn_voc.py) | VOC07 | VOC12 | Semi-Supervised |
| [semi\_supervised\_faster\_rcnn\_r50\_caffe\_fpn\_voc\_coco20.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/semi_supervised/semi_supervised_faster_rcnn_r50_caffe_fpn_voc_coco20.py) | VOC07 | VOC12 + COCO20 | Semi-Supervised |
| [faster\_rcnn\_r50\_caffe\_fpn\_coco.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/supervised/faster_rcnn_r50_caffe_fpn_coco.py) | COCO 2017 | N/A | Supervised |
| [semi\_supervised\_faster\_rcnn\_r50\_caffe\_fpn\_coco.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/semi_supervised/semi_supervised_faster_rcnn_r50_caffe_fpn_coco.py) | COCO 2017 | COCO 2017 | Semi-Supervised |
| [faster\_rcnn\_r50\_caffe\_fpn\_coco\_30shot.py](https://github.com/lexisnexis-risk-open-source/ledetection/configs/few_shot/faster_rcnn_r50_caffe_fpn_coco_30shot.py) | COCO 2017 | N/A | Few-Shot |

### Train on a Single GPU
We use the `tools/train.py` utility to launch model training on a single GPU. An example usage is as follows.

```bash
# Execute from project working directory `~/ledetection`.
python tools/train.py \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file. Recall that we created a new `./work_dirs` directory during our workspace setup, so in our configs, `work_dir = "work_dirs/"`.

The model can be configured to evaluate on the validation set every `interval` epoch or iteration, depending on whether `EpochBasedRunner` or `IterBasedRunner` is used. The evaluation interval can be specified in the config file as shown below.

```bash
# Assuming IterBasedRunner,
# evaluate the model every 4000 iterations.
evaluation = dict(interval=4000, metric="mAP")
```

### Train on CPU
To train on the CPU, for debugging purposes on machines without GPU, simply disable GPUs before launching the job.

```bash
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py
```

### Train on Multiple GPUs
We use `tools/dist_train.sh` to launch model training on multiple GPUs, leveraging efficient Distributed Data Parallelism (DDP). An example usage to train on 8 GPUs is as follows.

```bash
# Execute from project working directory `~/ledetection`.
bash tools/dist_train.sh \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py \
    8
```

### Example #1 - Train Supervised Faster R-CNN on PASCAL VOC
We provide two commands to train a supervised Faster R-CNN model on the PASCAL VOC dataset using our pre-built configuration.

```bash
# Within a Conda environment.
# Execute from project working directory `~/ledetection`.
bash tools/dist_train.sh \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py \
    8
```

```bash
# Using Docker.
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0,1,2,3,4,5,6,7"' \
    -u $(id -u) --rm --ipc=host \
    --env TORCH_HOME="data/torch/" \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    bash tools/dist_train.sh \
    configs/supervised/faster_rcnn_r50_caffe_fpn_voc07.py \
    8
```

For the Docker environment, we assume that the VOC dataset is located on the local machine at the path `/data/shared/`. We use the `docker run -v` flag to map volumes between the local host and the container at runtime. Our recommended best practice is to map two volumes from the local host to be used by the container:

1. We map the entire local LEDetection repository to the container so that any local modifications will take effect inside the container and can be used by the container at runtime.

    ```bash
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection
    ```

2. We map *data volumes* to a specified location inside the container so it can access data not previously copied during runtime.

    ```bash
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data
    ```

The default working directory of the container is set by `docker run -w ${APP_HOME_DIR}/ledetection`.

### Example #2 - Train Semi-Supervised Faster R-CNN on MS-COCO
We provide a similar `docker run` command to train a semi-supervised Faster R-CNN model on the MS-COCO 2017 dataset using our pre-built configuration.

```bash
# Using Docker.
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0,1,2,3,4,5,6,7"' \
    -u $(id -u) --rm --ipc=host \
    --env TORCH_HOME="data/torch/" \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    bash tools/dist_train_ssod.sh \
    configs/semi_supervised/semi_supervised_faster_rcnn_r50_caffe_fpn_coco.py \
    8 SoftERTeacher 10 1
```

Note that we use the modified `tools/dist_train_ssod.sh` script with additional arguments. The command above trains the Faster R-CNN model in semi-supervised mode using the `SoftERTeacher` protocol on 10% of labels sampled from the COCO `train2017` set.

## Testing
Using the `tools/dist_test.sh` script, we provide the following example `docker run` commands to evaluate trained models on standard VOC and COCO datasets. We assume the model checkpoints and config files exist in the working directory within `work_dirs`, and the data sources are located at the local path `/data/shared/`.

### Example #1 - Test Semi-Supervised Faster R-CNN on PASCAL VOC

```bash
# Test on VOC using the `mAP` metric averaged
# over ten overlap thresholds between 0.5 to 0.95.
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0,1,2,3,4,5,6,7"' \
    -u $(id -u) --rm --ipc=host \
    --env TORCH_HOME="data/torch/" \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    bash tools/dist_test.sh \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_voc/SoftERTeacher/semi_supervised_faster_rcnn_r50_caffe_fpn_voc.py \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_voc/SoftERTeacher/iter_18000.pth \
    8 --eval mAP \
    --eval-options iou_thr=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
```

### Example #2 - Test Semi-Supervised Faster R-CNN on MS-COCO

```bash
# Test on COCO using the `bbox` metric
# and print out per-class AP evaluations.
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0,1,2,3,4,5,6,7"' \
    -u $(id -u) --rm --ipc=host \
    --env TORCH_HOME="data/torch/" \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    bash tools/dist_test.sh \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_coco/SoftERTeacher/10/1/semi_supervised_faster_rcnn_r50_caffe_fpn_coco.py \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_coco/SoftERTeacher/10/1/iter_180000.pth \
    8 --eval bbox \
    --eval-options classwise=True
```

### Example #3 - Perform Inference on an Image File or Directory

```bash
# Perform inference from a config file and model checkpoint.
# The input can be path to an image file or directory of images.
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data/shared/
docker run \
    -w ${APP_HOME_DIR}/ledetection \
    --gpus='"device=0"' \
    -u $(id -u) --rm --ipc=host \
    --env TORCH_HOME="data/torch/" \
    -v ${LOCAL_HOME_DIR}/ledetection:${APP_HOME_DIR}/ledetection \
    -v ${DATA_DIR}:${APP_HOME_DIR}/ledetection/data \
    ledetection:pytorch1.11.0-cuda11.3 \
    python demo/image_demo.py \
    data/VOCdevkit/VOC2007/JPEGImages/ \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_voc/SoftERTeacher/semi_supervised_faster_rcnn_r50_caffe_fpn_voc.py \
    work_dirs/semi_supervised_faster_rcnn_r50_caffe_fpn_voc/SoftERTeacher/iter_18000.pth \
    --first-n 10 --outdir work_dirs/inference_results
```
