_base_ = "../_base_/supervised_lvis.py"
dataset_type = "LVISV1Dataset"
data_root = "data/coco/"
CLASSES = data_root + "annotations/lvis_v1_tail_classes337.txt"
pretrained = "torchvision://resnet50"
model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        frozen_stages=4,
        style="pytorch",
        init_cfg=dict(
            type="Pretrained", checkpoint=pretrained,
        ),
    ),
    neck=dict(frozen=True),
    rpn_head=dict(frozen=True),
    roi_head=dict(
        bbox_head=dict(
            frozen_layer_names=["shared_fcs"],
            num_classes=337,
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="ClassBalancedDataset",
            oversample_thr=1e-3,
            dataset=dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=data_root + "annotations/lvis_v1_train.json",
                img_prefix=data_root,
                filter_empty_gt=True,
            ),
        ),
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/lvis_v1_val.json",
        img_prefix=data_root,
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + "annotations/lvis_v1_val.json",
        img_prefix=data_root,
        filter_empty_gt=True,
    ),
)
model_type = "FasterRCNN"
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(_delete_=True, policy="fixed")
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20000)
evaluation = dict(interval=5000, metric="bbox")
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=10)
auto_resume = False
load_from = "results/lvis_v1_head866/${model_type}/r50/model_reset_remove.pth"
work_dir = "work_dirs/${cfg_name}/${model_type}/"

