import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch

import mmcv
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes

from ledet.datasets import build_dataset
from ledet.utils import patch_config
from ledet.models.utils import Transform2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse a training dataset. "
        "Example usage: "
        "python ./browse_dataset.py "
        "configs/semi_supervised/semi_supervised_faster_rcnn_r50_caffe_fpn_coco.py "
        "--first-n 5 --cfg-options percent=1 fold=1 "
        "--output-dir work_dirs/coco_dataset_vis --not-show"
    )
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--first-n",
        type=int,
        default=20,
        help="View first N files."
    )
    parser.add_argument(
        "--skip-type",
        type=str,
        nargs="+",
        default=["DefaultFormatBundle", "Normalize", "Collect"],
        help="Skip unused pipelines."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="If there is no display interface, save to file."
    )
    parser.add_argument(
        "--not-show", default=False, action="store_true"
    )
    parser.add_argument(
        "--show-interval",
        type=float,
        default=2,
        help="The interval of show (second)."
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        "be overwritten is a list, it should be like key='[a,b]' or key=a,b "
        "It also allows nested list/tuple values, e.g. key='[(a,b),(c,d)]' "
        "Note that the quotation marks are necessary and that no white space "
        "is allowed."
    )
    args = parser.parse_args()
    return args


def skip_pipeline(pipelines, skip_type):
    if isinstance(pipelines, list):
        new_pipelines = []
        for pipe in pipelines:
            pipe = skip_pipeline(pipe, skip_type)
            if pipe is not None:
                new_pipelines.append(pipe)
        return new_pipelines
    elif isinstance(pipelines, dict):
        if pipelines["type"] in skip_type:
            return None
        elif pipelines["type"] == "MultiBranch":
            new_pipelines = {}
            for k, v in pipelines.items():
                if k != "type":
                    new_pipelines[k] = skip_pipeline(v, skip_type)
                else:
                    new_pipelines[k] = v
            return new_pipelines
        else:
            return pipelines
    else:
        raise NotImplementedError()


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    
    def extract_dataset_cfg(cfg):
        if "dataset" in cfg:
            cfg = cfg["dataset"]
        return cfg
    
    cfg = Config.fromfile(config_path)
    # Set `cfg.work_dir` to avoid AttributeError, but it won't be used.
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    train_data_cfg = extract_dataset_cfg(train_data_cfg)
    if train_data_cfg["type"] == "SemiDataset":
        for dset in ["sup", "unsup"]:
            train_data_cfg[dset] = extract_dataset_cfg(train_data_cfg[dset])
            train_data_cfg[dset]["pipeline"] = skip_pipeline(train_data_cfg[dset]["pipeline"], skip_type)
    else:
        train_data_cfg["pipeline"] = skip_pipeline(train_data_cfg["pipeline"], skip_type)
    return cfg


def browse(dataset, dataset_type, args):
    text_color = "cyan"
    bbox_color = "cyan"
    progress_bar = mmcv.ProgressBar(args.first_n)
    for item in dataset:
        bboxes = []
        labels = []
        trans_mats = []
        out_shapes = []
        for it in item:
            filepath = (
                os.path.join(args.output_dir, dataset_type, "sup_" + Path(it["filename"]).name)
                if args.output_dir is not None
                else None
            )
            gt_masks = it.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = mask2ndarray(gt_masks)
            if "tag" not in it or it["tag"].startswith("sup"):
                # Supervised mmdet or ledet dataset
                imshow_det_bboxes(
                    it["img"],
                    it["gt_bboxes"],
                    it["gt_labels"],
                    gt_masks,
                    class_names=args.CLASSES,
                    show=not args.not_show,
                    wait_time=args.show_interval,
                    out_file=filepath,
                    bbox_color=bbox_color,
                    text_color=text_color,
                )
                progress_bar.update()
            else:
                # Semi-supervised multi-branch ledet dataset
                trans_mats.append(it["transform_matrix"])
                bboxes.append(it["gt_bboxes"])
                labels.append(it["gt_labels"])
                out_shapes.append(it["img_shape"])
                if len(trans_mats) == 2:
                    # check equality between different augmentation
                    trans_bboxes = Transform2D.transform_bboxes(
                        torch.from_numpy(bboxes[1]).float(),
                        torch.from_numpy(trans_mats[0]).float()
                        @ torch.from_numpy(trans_mats[1]).float().inverse(),
                        out_shapes[0],
                    )
                    img = imshow_det_bboxes(
                        item[0]["img"],
                        item[0]["gt_bboxes"],
                        item[0]["gt_labels"],
                        class_names=args.CLASSES,
                        show=False,
                        wait_time=args.show_interval,
                        out_file=filepath.replace("sup", "unsup_student"),
                        bbox_color=bbox_color,
                        text_color=bbox_color,
                    )
                    imshow_det_bboxes(
                        item[1]["img"],
                        item[1]["gt_bboxes"],
                        item[1]["gt_labels"],
                        class_names=args.CLASSES,
                        show=False,
                        wait_time=args.show_interval,
                        out_file=filepath.replace("sup", "unsup_teacher"),
                        bbox_color="green",
                        text_color="green",
                    )
                    imshow_det_bboxes(
                        img,
                        trans_bboxes.numpy(),
                        labels[1],
                        class_names=args.CLASSES,
                        show=not args.not_show,
                        wait_time=args.show_interval,
                        out_file=filepath.replace("sup", "overlay"),
                        bbox_color="green",
                        text_color="green",
                        thickness=5,
                    )

    
def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    if cfg.data.train.type == "SemiDataset":
        dataset_type = "semi"
    else:
        dataset_type = "sup"
    dataset = build_dataset(cfg.data.train)
    args.CLASSES = dataset.CLASSES
    
    sup_data = []
    semi_data = []
    data_list = []
    break_flag = False
    for item in dataset:
        if break_flag:
            break
        if not isinstance(item, list):
            item = [item]
        for it in item:
            if dataset_type == "semi":
                if it["tag"].startswith("sup"):
                    if len(sup_data) < args.first_n:
                        sup_data.append(item)
                else:
                    if item not in semi_data:
                        semi_data.append(item)
                data_list = sup_data + semi_data
                if len(data_list) == 2 * args.first_n:
                    break_flag = True
            else:
                data_list.append(item)
                if len(data_list) == args.first_n:
                    break_flag = True

    browse(data_list, dataset_type, args)
    

if __name__ == "__main__":
    main()
