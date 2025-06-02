_base_ = "../_base_/semi_supervised_lvis_augmented_head.py"
dataset_type = "LVISV1Dataset"
data_root = "data/coco/"
CLASSES = data_root + "annotations/lvis_v1_head_classes866.txt"
pretrained = "torchvision://resnet50"
model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(requires_grad=True),
        norm_eval=True,
        frozen_stages=1,
        style="pytorch",
        init_cfg=dict(
            type="Pretrained", checkpoint=pretrained,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=866,
        )
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300,
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
        unsup_weight_beta=1.0,
        unsup_weight_warmup=30000,
        debug_mode=False,
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
    samples_per_gpu=8, # 8x GPU training
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type="MultiImageMixDataset",
            dataset=dict(
                type="ClassBalancedDataset",
                oversample_thr=1e-3,
                dataset=dict(
                    type=dataset_type,
                    classes=CLASSES,
                    ann_file=data_root + "annotations/lvis_v1_train.json",
                    img_prefix=data_root, # LVIS v1 - image prefix specified in filename
                    filter_empty_gt=True,
                )
            ),
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file=data_root + "annotations/instances_unlabeled2017.json",
            img_prefix=data_root + "unlabeled2017/",
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/lvis_v1_val.json",
        img_prefix=data_root, # LVIS v1 - image prefix specified in filename
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/lvis_v1_val.json",
        img_prefix=data_root, # LVIS v1 - image prefix specified in filename
        filter_empty_gt=True,
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
            epoch_length=7330,
        )
    ),
)
model_type = "SoftERTeacher"
evaluation = dict(type="SubModulesDistEvalHook", metric="bbox", interval=60000 * 3)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 3, 160000 * 3])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 3)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=100)
auto_resume = False
work_dir = "work_dirs/${cfg_name}/${model_type}/"
