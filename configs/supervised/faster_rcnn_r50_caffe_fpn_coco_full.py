_base_ = "../_base_/supervised_coco.py"
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "annotations/instances_train2017.json",
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
model_type = "FasterRCNN"
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)
evaluation = dict(interval=4000 * 4, metric="bbox")
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
work_dir = "work_dirs/${cfg_name}/${model_type}"

