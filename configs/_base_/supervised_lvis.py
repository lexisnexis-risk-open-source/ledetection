mmdet_base = "../../../mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/models/faster_rcnn_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_detection.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Sequential",
        transforms=[
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
]
train_pipeline = [
    dict(type="CopyPaste", max_num_pasted=100),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type="MultiImageMixDataset",
        dataset=dict(
            type="ClassBalancedDataset",
            oversample_thr=1e-3,
            dataset=dict(
                type="LVISV1Dataset",
                ann_file=None,
                img_prefix=None,
                pipeline=load_pipeline,
                filter_empty_gt=True,
            ),
        ),
        pipeline=train_pipeline,
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
