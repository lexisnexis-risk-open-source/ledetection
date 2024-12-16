_base_ = "../_base_/supervised_lvis.py"
dataset_type = "LVISV1Dataset"
data_root = "data/coco/"
CLASSES = data_root + "annotations/lvis_v1_all_classes1203.txt"
pretrained = "torchvision://resnet101"
model = dict(
    backbone=dict(
        depth=101,
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
            num_classes=1203,
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="ClassBalancedDataset",
            oversample_thr=1e-3,
            dataset=dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=data_root + "annotations/lvis_v1_train_${fold}@${shot}.json",
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
fold = "seed1"
shot = "30shots"
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[60000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=80000)
evaluation = dict(interval=20000, metric="bbox")
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
auto_resume = False
load_from = "results/lvis_v1_head866/${model_type}/r101/model_reset_combine.pth"
work_dir = "work_dirs/${cfg_name}/${model_type}/${shot}/${fold}/"
