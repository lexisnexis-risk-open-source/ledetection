_base_ = "../_base_/supervised_coco.py"
dataset_type = "CocoDataset"
data_root = "data/coco/"
CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        frozen_stages=4,
        style="caffe",
        depth=101,
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet101_caffe"
        ),
    ),
    neck=dict(frozen=True),
    rpn_head=dict(frozen=True),
    roi_head=dict(
        bbox_head=dict(
            frozen_layer_names=["shared_fcs", "fc_reg"],
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
            ann_file=[
                data_root + "annotations/few_shot/cocosplit2017/${fold}/full_box_${shot}_" + c + "_trainval.json" for c in CLASSES
            ],
            img_prefix=data_root + "train2017/",
            filter_empty_gt=True,
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        filter_empty_gt=True,
    ),
)
fold = "seed1"
shot = "5shot"
model_type = "SoftERTeacher"
evaluation = dict(interval=4000 * 2, metric="bbox")
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[72000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
load_from = "results/coco_few_shot_base60/coco2017/${model_type}/r101/model_reset_combine.pth"
work_dir = "work_dirs/${cfg_name}/${model_type}/${shot}/${fold}/"

