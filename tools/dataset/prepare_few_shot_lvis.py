import argparse
import json
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/coco/annotations/lvis_v1_train.json",
        help="path to the annotation file",
    )
    parser.add_argument(
        "--shots", type=int, default=30, help="number of shots"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed number"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory to save annotation file",
    )
    args = parser.parse_args()
    return args


def get_shots(args):
    dataset = json.load(open(args.data_path, "r"))
    cat_ids = {cat["id"]: cat["name"] for cat in dataset["categories"]}
    all_anns = dataset["annotations"]
    anno_cat = defaultdict(list)
    ann_counts = defaultdict(int)
    for a in all_anns:
        anno_cat[a["category_id"]].append(a)
    
    sampled_anns = []
    for ID, name in cat_ids.items():
        if len(anno_cat[ID]) <= args.shots:
            shots = anno_cat[ID]
        else:
            shots = random.sample(anno_cat[ID], args.shots)
        sampled_anns.extend(shots)

    new_data = {
        "info": dataset["info"],
        "licenses": dataset["licenses"],
        "categories": dataset["categories"],
        "images": dataset["images"],
        "annotations": sampled_anns,
    }

    save_name = args.data_path.split("/")[-1]
    save_name = os.path.splitext(save_name)[0] + \
        "_seed{}@{}shots.json".format(args.seed, args.shots)
    save_path = os.path.join(args.save_dir, save_name)
    with open(save_path, "w") as f:
        print(f"Saving annotation file to {save_path}")
        json.dump(new_data, f)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    if not args.save_dir:
        # By default, save to directory of `data_path`
        args.save_dir = os.path.dirname(args.data_path)
    os.makedirs(args.save_dir, exist_ok=True)
    get_shots(args)
