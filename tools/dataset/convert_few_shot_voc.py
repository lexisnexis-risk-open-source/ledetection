"""Example output:
vocsplit/       # The data root.
|-- seed1/      # Subdirectory.
|   |-- box_10shot_2007_all_train.txt  # Contains all 2007 classes with 10 shots.
|   |-- box_10shot_2012_all_train.txt  # Contains all 2012 classes with 10 shots.
|   |-- etc...
|-- seed2/
|   |-- etc...
"""
import os
import os.path as osp
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="First, recursively download the contents of `vocsplit` from "
        "http://dl.yf.io/fs-det/datasets/vocsplit/ "
        "into local directory, e.g., `data/VOCdevkit/few_shot/vocsplit/`. "
        "Then, run this script to convert all training samples from each shot "
        "under each seed subdirectory into corresponding single files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/VOCdevkit/few_shot/vocsplit/",
        help="Path to input data source."
    )
    parser.add_argument(
        "--shots",
        type=str,
        nargs="+",
        default=["1shot", "2shot", "3shot", "5shot", "10shot"],
        help="Number of bounding box instances."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Save directory. Default to same directory as `--data-root`."
    )
    args = parser.parse_args()
    return args


def convert_voc(args):
    for dirname in os.listdir(args.data_root):
        subdir = osp.join(args.data_root, dirname)
        if osp.isdir(subdir):
            if args.save_dir:
                savedir = osp.join(args.save_dir, dirname)
                os.makedirs(savedir, exist_ok=True)
            else:
                savedir = subdir
            shot2path = defaultdict(list)
            filenames = [name for name in os.listdir(subdir) if name.endswith(".txt")]
            for shot in args.shots:
                voc07_all_ids = []
                voc12_all_ids = []
                for filename in filenames:
                    if shot in filename:
                        filepath = osp.join(subdir, filename)
                        with open(filepath, "r") as f:
                            for line in f:
                                shot2path[shot].append(line.strip())
                for line in shot2path[shot]:
                    image_id = line.split("/")[-1].replace(".jpg", "")
                    if "VOC2007" in line:
                        voc07_all_ids.append(image_id)
                    elif "VOC2012" in line:
                        voc12_all_ids.append(image_id)
                voc07_out = osp.join(savedir, f"box_{shot}_2007_all_train.txt")
                voc12_out = osp.join(savedir, f"box_{shot}_2012_all_train.txt")
                for out in [voc07_out, voc12_out]:
                    with open(out, "w") as f:
                        if "2007" in out:
                            for line in voc07_all_ids:
                                f.write(line + "\n")
                        elif "2012" in out:
                            for line in voc12_all_ids:
                                f.write(line + "\n")


if __name__ == "__main__":
    args = parse_args()
    convert_voc(args)
