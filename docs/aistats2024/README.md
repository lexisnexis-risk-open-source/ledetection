# AISTATS 2024
This section provides a guide to (closely) reproduce the results reported in Tables 1-3 of our [AISTATS 2024 paper](https://arxiv.org/abs/2303.05739).

## Environment
Follow the installation and quickstart [documentation](https://github.com/lexisnexis-risk-open-source/ledetection/tree/main/docs) to build the prerequisite environment. We used MMDetection v2.16.0 to produce our results. However, we also tested using MMDetection v2.28.0 with similar results. Either version will work fine, but v2.28.0 comes with the benefit of faster training. At this time, we do not yet support MMDetection v3.x.

All experiments train on 8x GPUs. We used NVIDIA RTX A6000 with 48GB of video memory. You may need to adjust the GPU-batchsize configuration according to your GPU budget.

## Data Preparation
We assume the required COCO and VOC datasets are located at the respective paths `data/coco/` and `data/VOCdevkit/`, which are relative to the LEDetection repository. For the exact data splits related to semi-supervised and few-shot experiments, you can download from our archival at [https://zenodo.org/doi/10.5281/zenodo.8007045](https://zenodo.org/doi/10.5281/zenodo.8007045).

You can also re-create your own data partitions using the provided scripts in `tools/dataset/`. However, note that randomness may still occur with different Python package versions, even when we provided a specific random seed, so your data splits may not be exactly the same as ours.

## Few-Shot Procedure
Our approach to semi-supervised few-shot detection follows a simple multi-stage procedure. The steps below should reproduce the results in Tables 1 and 2 of our paper.

```
1. Base pre-training --> 2. Novel weights initialization --> 3. Few-shot fine-tuning
```

**Step 1: Base Pre-Training**

We pre-train a detector on base classes using the combination of base exemplars and available unlabeled images. Below are some example training commands and configurations. See the [few-shot configs](https://github.com/lexisnexis-risk-open-source/ledetection/tree/main/configs/few_shot) for the full details.

<details open>
<summary>VOC</summary>

```bash
#### Supervised base pre-training
SPLIT=part1            # repeat with `part2` and `part3`
bash tools/dist_train.sh \
    configs/few_shot/faster_rcnn_r50_caffe_fpn_voc0712_base15_${SPLIT}.py \
    8                  # number of GPUs

#### Semi-supervised base pre-training
BACKBONE=r50           # repeat with `r101`
SPLIT=part1            # repeat with `part2` and `part3`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
bash tools/dist_train_ssod.sh \
    configs/few_shot/semi_supervised_faster_rcnn_${BACKBONE}_caffe_fpn_voc0712_base15_${SPLIT}.py \
    8 \                # number of GPUs
    ${MODEL} \         # model type
    NA NA              # extra arguments not used
```

</details>


<details open>
<summary>COCO</summary>

```bash
#### Supervised base pre-training
bash tools/dist_train.sh \
    configs/few_shot/faster_rcnn_r50_caffe_fpn_coco_base60.py \
    8

#### Semi-supervised base pre-training
BACKBONE=r50           # repeat with `r101`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
bash tools/dist_train_ssod.sh \
    configs/few_shot/semi_supervised_faster_rcnn_${BACKBONE}_caffe_fpn_coco_base60.py \
    8 \                # number of GPUs
    ${MODEL} \         # model type
    NA NA              # extra arguments not used
```

</details>

**Step 2: Novel Weights Initialization**

We initialize the weights of the novel classes in two ways, depending on the dataset.

<details open>
<summary>VOC</summary>

We initialize the VOC novel classes to random values while reusing the weights of the base classes.

```bash
SPLIT=part1            # repeat with `part2` and `part3`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
python tools/misc/ckpt_surgery.py \
    --src1 results/voc0712_base15/${SPLIT}/${MODEL}/iter_60000.pth \
    --method randinit
```

where `--src1` is the path to the weights of the base model obtained from Step 1. By default, the resulting combined weights, pre-trained base weights + random novel weights, are saved to the same directory as `--src1`. For example, `results/voc0712_base15/${SPLIT}/${MODEL}/model_reset_surgery.pth`.

</details>


<details open>
<summary>COCO</summary>

First, we initialize the COCO novel classes to random values by removing the head `cls` and `reg` layers.

```bash
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
python tools/misc/ckpt_surgery.py \
    --src1 results/coco_few_shot_base60/coco2017/${MODEL}/iter_720000.pth \
    --method remove \
    --keep-student-teacher --coco
```

where `--src1` is the path to the weights of the base model obtained from Step 1. The resulting modified weights, without the `cls` and `reg` layers, are saved to `results/coco_few_shot_base60/coco2017/${MODEL}/model_reset_remove.pth`.

Then, we train the novel detector from scratch, on novel classes, while freezing the rest of the architecture to preserve the weights of the base classes.

```bash
set -x
BACKBONE=r50           # repeat with `r101`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
for SHOT in 30shot 10shot 5shot 1shot; do
    for FOLD in seed1 seed2 seed3 seed4 seed5; do
        bash tools/dist_train_fsod.sh \
            configs/few_shot/semi_supervised_faster_rcnn_${BACKBONE}_caffe_fpn_coco_novel20.py \
            8 \
            ${MODEL} ${SHOT} ${FOLD}
        sleep 1m
    done
done
```

</details>


**Step 3: Few-Shot Fine-Tuning**

We combine the weights of the base and novel models obtained in the previous Steps 1 and 2. Finally, we fine-tune the few-shot detector on a balanced training set of $k$ shots per class containing both base and novel instances.

<details open>
<summary>VOC</summary>

During few-shot fine-tuning on VOC, we update both the RoI box classifier and regressor while freezing all other components.

```bash
set -x
BACKBONE=r50           # repeat with `r101`
SPLIT=part1            # repeat with `part2` and `part3`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
for SHOT in 1shot 5shot 10shot; do
    for FOLD in seed1 seed2 seed3 seed4 seed5 seed6 seed7 seed8 seed9 seed10; do
        bash tools/dist_train_fsod.sh \
            configs/few_shot/faster_rcnn_${BACKBONE}_caffe_fpn_voc0712_${SHOT}_${SPLIT}.py \
            8 \
            ${MODEL} ${SHOT} ${FOLD}
        sleep 1m
    done
done
```

</details>


<details open>
<summary>COCO</summary>

During few-shot fine-tuning on COCO, we update only the RoI box classifier while freezing all other components, including the box regressor.

```bash
set -x
BACKBONE=r50           # repeat with `r101`
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
for SHOT in 30shot 10shot 5shot 1shot; do
    for FOLD in seed1 seed2 seed3 seed4 seed5; do
        ## Combine the weights of the base and novel models.
        python tools/misc/ckpt_surgery.py \
            --src1 results/coco_few_shot_base60/coco2017/${MODEL}/iter_720000.pth \
            --src2 results/coco_few_shot_novel20/coco2017/${MODEL}/${SHOT}/${FOLD}/iter_48000.pth \
            --method combine \
            --coco
        sleep 1m
        ## Few-shot fine-tuning.
        bash tools/dist_train_fsod.sh \
            configs/few_shot/faster_rcnn_${BACKBONE}_caffe_fpn_coco_${SHOT}.py \
            8 \
            ${MODEL} ${SHOT} ${FOLD}
        sleep 1m
    done
done
```

where `--src2` is the path to the weights of the novel model obtained in Step 2. The resulting combined weights, of both base and novel models, are saved to `results/coco_few_shot_base60/coco2017/${MODEL}/model_reset_combine.pth`.

</details>


## Semi-Supervised Few-Shot Procedure
To reproduce the results in Table 3 of our paper, follow the steps below, which are similar to the Few-Shot Procedure described above. For semi-supervised few-shot detection, the base classes are constrained to be limited in quantity at {1, 5, 10} percent. These steps are performed only on the COCO dataset using the ResNet-101 backbone.

**Step 1: Semi-Supervised Base Pre-Training**

```bash
set -x
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
for PERCENT in 1 5 10; do
    for FOLD in 1 2 3 4 5; do
        bash tools/dist_train_ssod.sh \
            configs/semi_few_shot/semi_supervised_faster_rcnn_r101_caffe_fpn_coco_semi_few_base60.py \
            8 \
            ${MODEL} ${PERCENT} ${FOLD}
        sleep 1m
    done
done
```

**Step 2: Semi-Supervised Novel Weights Initialization**

We remove the head `cls` and `reg` layers of the base model, initialize them with random values, and train them from scratch using a combination of novel few-shot classes and unlabeled images. The weights of the base model are reused and frozen during novel class initialization (training).

```bash
set -x
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
PERCENT=10             # repeat with {1,5} percent
for SHOT in 30shot 10shot 5shot; do
    for FOLD in 1 2 3 4 5; do
        ## Remove the head `cls` and `reg` layers.
        python tools/misc/ckpt_surgery.py \
            --src1 results/coco_semi_few_base60/${MODEL}/${PERCENT}/${FOLD}/iter_180000.pth \
            --keep-student-teacher --method remove --coco
        sleep 1m
        ## Train head layers initialized with random values on novel classes.
        bash tools/dist_train_semi_few.sh \
            configs/semi_few_shot/semi_supervised_faster_rcnn_r101_caffe_fpn_coco_semi_few_novel20.py \
            8 \
            ${MODEL} ${PERCENT} ${FOLD} ${SHOT} seed${FOLD}
        sleep 1m
    done
done
```

**Step 3: Few-Shot Fine-Tuning**

We combine the weights of the base and novel models obtained in the previous Steps 1 and 2. Finally, we fine-tune the few-shot detector on a balanced training set of $k$ shots per class containing both base and novel instances. We update only the RoI box classifier while freezing all other components, including the box regressor.

```bash
set -x
MODEL=SoftERTeacher    # repeat with `SoftTeacher`
PERCENT=10             # repeat with {1,5} percent
for SHOT in 30shot 10shot 5shot; do
    for FOLD in 1 2 3 4 5; do
        ## Combine the weights of the base and novel models.
        python tools/misc/ckpt_surgery.py \
            --src1 results/coco_semi_few_base60/${MODEL}/${PERCENT}/${FOLD}/iter_180000.pth \
            --src2 results/coco_semi_few_novel20/${MODEL}/${PERCENT}/${FOLD}/${SHOT}/seed${FOLD}/iter_48000.pth \
            --method combine --coco
        sleep 1m
        ## Few-shot fine-tuning.
        bash tools/dist_train_semi_few.sh \
            configs/semi_few_shot/faster_rcnn_r101_caffe_fpn_coco_semi_${SHOT}.py \
            8 \
            ${MODEL} ${PERCENT} ${FOLD} ${SHOT} seed${FOLD}
        sleep 1m
    done
done
```
