_base_ = "../_base_/supervised_lvis.py"
dataset_type = "LVISV1Dataset"
data_root = "data/coco/"
# The ``CLASSES`` file includes a misspelling that came with the official LVIS v1 dataset.
# e.g., ``speaker_(stero_equipment)``
CLASSES = data_root + "annotations/lvis_v1_all_classes1203.txt"
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
            # LVIS v1 has 1203 classes
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
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 3, 160000 * 3])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 3)
evaluation = dict(interval=60000 * 3, metric="bbox")
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
auto_resume = False
work_dir = "work_dirs/${cfg_name}/${model_type}/"

