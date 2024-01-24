_base_ = "../_base_/semi_supervised_voc.py"
dataset_type = "VOCDataset"
data_root = "data/VOCdevkit/"
CLASSES = (
    "aeroplane", "bicycle", "boat", "bottle", "car", "cat",
    "chair", "diningtable", "dog", "horse", "person", "pottedplant",
    "sheep", "train", "tvmonitor"
)
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        frozen_stages=1,
        style="caffe",
        depth=101,
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet101_caffe"
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(CLASSES),
        )
    ),
)
semi_wrapper = dict(
    type="${model_type}",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_box_size=0,
        unsup_weight_alpha=2.0,
        unsup_weight_beta=4.0,
        unsup_weight_warmup=1000,
        sim_cls_loss=dict(
            type="CrossEntropySimilarityLoss",
            reduction="mean",
            loss_weight=1.0),
        iou_bbox_loss=dict(
            type="IoULoss",
            reduction="mean",
            loss_weight=1.0)),
    test_cfg=dict(inference_on="teacher"),
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        sup=dict(
            type="RepeatDataset",
            times=3,
            dataset=dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=[
                    data_root + "VOC2007/ImageSets/Main/trainval.txt",
                    data_root + "VOC2012/ImageSets/Main/trainval.txt"
                ],
                img_prefix=[data_root + "VOC2007/", data_root + "VOC2012/"],
                filter_empty_gt=True,
            )
        ),
        unsup=dict(
            type="RepeatDataset",
            times=1,
            dataset=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/instances_train2017.json",
                img_prefix="data/coco/train2017/",
                filter_empty_gt=False,
            )
        )
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        filter_empty_gt=True,
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
            epoch_length=7330,
        )
    ),
)
model_type = "SoftTeacher"
evaluation = dict(type="SubModulesDistEvalHook", interval=4000, metric="mAP")
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[40000, 52000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
work_dir = "work_dirs/${cfg_name}/${model_type}"
