import json
import argparse
import imagesize
import os.path as osp
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate unlabeled JSON dataset "
        "from a directory of images following the COCO format. "
        "Example usage: "
        "python tools/dataset/generate_unlabeled_json.py "
        "--data-root data/coco/unlabeled2017/ "
        "--save-path data/coco/annotations/unlabeled_images.json"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/coco/unlabeled2017/",
        help="Path to unlabeled data source."
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Whether input `--data-root` is nested with subdirectories."
    )
    parser.add_argument(
        "--include-ext",
        type=str,
        nargs="+",
        default=[
            ".jpg", ".jpeg", ".png",
            ".JPG", ".JPEG", ".PNG"
        ],
        help="Filename extensions to include."
    )
    parser.add_argument(
        "--exclude-subdir",
        type=str,
        nargs="+",
        default=[],
        help="Subdirectories to exclude. "
        "Multiple entries separated by blank space. "
        "No quotations around entries."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Path to save output JSON. Default to input root directory."
    )
    args = parser.parse_args()
    return args


def nested_directory_to_json(path, args):
    images = []
    for subdir in path.rglob("*"):
        if osp.isdir(subdir) and not subdir.name.startswith("."):
            if subdir.name not in args.exclude_subdir:
                filepaths = get_file_paths(subdir, args)
                sub_images = jsonify(subdir.name, filepaths)
                images.extend(sub_images)
    assert len(images) > 0, \
        f"No valid images in `{path}`."
    save_json(images, args.save_path)


def directory_to_json(path, args):
    filepaths = get_file_paths(path, args)
    images = jsonify("", filepaths)
    save_json(images, args.save_path)


def get_file_paths(path, args):
    valid_paths = [path.glob("*" + e) for e in args.include_ext]
    filepaths = [filepath for p in valid_paths for filepath in p]
    assert len(filepaths) > 0, \
        f"Directory `{path}` is empty."
    return filepaths

    
def jsonify(subdir, file_paths):
    images = []
    for path in file_paths:
        w, h = imagesize.get(path)
        img_id = osp.splitext(path.name)[0]
        filename = str(path).split("/")[-1]
        filename = osp.join(subdir, filename)
        per_image_dict = dict(
            id=img_id,
            file_name=filename,
            width=w,
            height=h
        )
        images.append(per_image_dict)
    return images


def save_json(images, save_path):
    data = dict(categories=[])
    data["images"] = images
    with open(save_path, "w") as f:
        print(f"Saving output to {save_path}")
        json.dump(data, f)


if __name__ == "__main__":
    args = parse_args()
    if not args.save_path:
        # By default, save output to root directory of input data.
        args.save_path = "/".join(args.data_root.split("/")[:-2])
        args.save_path = osp.join(args.save_path, "unlabeled.json")
    path = Path(args.data_root)
    if args.nested:
        nested_directory_to_json(path, args)
    else:
        directory_to_json(path, args)