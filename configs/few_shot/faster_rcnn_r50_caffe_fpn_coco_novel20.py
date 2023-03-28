_base_ = "../_base_/supervised_coco.py"
dataset_type = "CocoDataset"
data_root = "data/coco/"
CLASSES = (
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "dining table", "dog", "horse", "motorcycle", "person", "potted plant",
    "sheep", "couch", "train", "tv"
)
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        frozen_stages=4,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ),
    neck=dict(frozen=True),
    rpn_head=dict(frozen=True),
    roi_head=dict(
        bbox_head=dict(
            frozen_layer_names=["shared_fcs"],
            num_classes=len(CLASSES),
        )
    ),
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type=dataset_type,
            classes=CLASSES,
            ann_file=[
                data_root + "annotations/few_shot/cocosplit2017/${fold}/full_box_${shot}_" + c + "_trainval.json" for c in CLASSES
            ],
            img_prefix=data_root + "train2017/",
            filter_empty_gt=True,
        )
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        filter_empty_gt=True,
    ),
)
fold = "seed1"
shot = "30shot"
model_type = "FasterRCNN"
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy="fixed")
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=10000)
evaluation = dict(interval=1000, metric="bbox")
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
load_from = "results/coco_few_shot_base60/coco2017/${model_type}/model_reset_remove.pth"
work_dir = "work_dirs/${cfg_name}/${model_type}/${shot}/${fold}/"

