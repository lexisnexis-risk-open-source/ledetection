# Copyright (c) OpenMMLab. All rights reserved.
# Modified from mmdetection/demo/image_demo.py
import os
from pathlib import Path
from argparse import ArgumentParser

from mmcv import Config
from mmdet.apis import (
    show_result_pyplot,
    inference_detector
)

from ledet.apis.inference import (
    save_result,
    init_detector
)
from ledet.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img_file_or_dir", help="Image file or directory.")
    parser.add_argument("config", help="Config file.")
    parser.add_argument("checkpoint", help="Checkpoint file.")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference.")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="bbox score threshold."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Specify the directory to save visualization results.",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=10,
        help="Process first N files for inference.",
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    # Set `cfg.work_dir` to avoid AttributeError, but it won't be used.
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # Build the model from config and checkpoint file.
    model = init_detector(cfg, args.checkpoint, device=args.device)
    path = Path(args.img_file_or_dir)
    if not path.exists():
        raise OSError(
            "Path to image file or directory does not exist. "
            f"Got `{path}`"
        )
    if path.is_dir():
        imgs = path.glob("*")
        if args.first_n >= 0:
            imgs = [
                img for count,img in enumerate(imgs) if count < args.first_n
            ]
    else:
        imgs = [path]
    for img in imgs:
        # Test a single image.
        result = inference_detector(model, img)
        # Show the result.
        if args.outdir is None:
            show_result_pyplot(model, img, result, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.outdir, os.path.basename(img))
            print(f"Save result to {out_file_path}")
            save_result(
                model, img, result, score_thr=args.score_thr, out_file=out_file_path
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
