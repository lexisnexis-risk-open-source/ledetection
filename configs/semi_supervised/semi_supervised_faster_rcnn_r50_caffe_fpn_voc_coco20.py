_base_ = "base_voc_coco20.py"
data_root = "data/VOCdevkit/"
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        frozen_stages=1,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
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
        min_pseduo_box_size=0,
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
                type="VOCDataset",
                ann_file=[
                    data_root + "VOC2007/ImageSets/Main/trainval.txt",
                ],
                img_prefix=[data_root + "VOC2007/"],
                filter_empty_gt=True,
            )
        ),
    ),
    val=dict(
        type="VOCDataset",
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        filter_empty_gt=True,
    ),
    test=dict(
        type="VOCDataset",
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
evaluation = dict(type="SubModulesDistEvalHook", interval=3000, metric="mAP")
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[36000, 48000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=54000)
checkpoint_config = dict(by_epoch=False, interval=3000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
work_dir = "work_dirs/${cfg_name}/${model_type}"




