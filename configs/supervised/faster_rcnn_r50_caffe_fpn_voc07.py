_base_ = "base_voc.py"
dataset_type = "VOCDataset"
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + "VOC2007/ImageSets/Main/trainval.txt",
            ],
            img_prefix=[data_root + "VOC2007/"],
            filter_empty_gt=True,
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        filter_empty_gt=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        filter_empty_gt=True,
    ),
)
model_type = "FasterRCNN"
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[12000, 16000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=18000)
evaluation = dict(interval=4000, metric="mAP")
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
auto_resume = False
fp16 = dict(loss_scale="dynamic")
work_dir = "work_dirs/${cfg_name}/${model_type}"



