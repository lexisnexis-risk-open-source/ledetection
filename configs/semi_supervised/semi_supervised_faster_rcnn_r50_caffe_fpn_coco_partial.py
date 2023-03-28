_base_ = "../_base_/semi_supervised_coco.py"
dataset_type = "CocoDataset"
data_root = "data/coco/"
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
        unsup_weight_alpha=4.0,
        unsup_weight_beta=8.0,
        unsup_weight_warmup=10000,
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
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="RepeatDataset",
            times=3,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root + "annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
                img_prefix=data_root + "train2017/",
                filter_empty_gt=True,
            )
        ),
        unsup=dict(
            type=dataset_type,
            ann_file=data_root + "annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix=data_root + "train2017/",
            filter_empty_gt=False,
        ),
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
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
            epoch_length=7330,
        )
    ),

)
fold = 1
percent = 1
model_type = "SoftTeacher"
evaluation = dict(type="SubModulesDistEvalHook", metric="bbox", interval=4000)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000, 160000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
work_dir = "work_dirs/${cfg_name}/${model_type}/${percent}/${fold}"

