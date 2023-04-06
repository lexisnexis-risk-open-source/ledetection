# Modified from
# https://github.com/ucbdrive/few-shot-object-detection/blob/master/tools/ckpt_surgery.py
import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--src1", type=str, default="", help="Path to the main checkpoint."
    )
    parser.add_argument(
        "--src2",
        type=str,
        default="",
        help="Path to the secondary checkpoint (for combining).",
    )
    parser.add_argument(
        "--save-dir", type=str, default="", help="Save directory."
    )
    # Surgery method
    parser.add_argument(
        "--method",
        choices=["combine", "remove", "randinit"],
        required=True,
        help="Surgery method. `combine`: "
        "combine checkpoints. `remove`: for fine-tuning on "
        "novel dataset, remove the final layer of the "
        "base detector. `randinit`: randomly initialize "
        "novel weights.",
    )
    parser.add_argument(
        "--keep-student-teacher",
        action="store_true",
        help="Perform student-teacher multi-branch surgery."
    )
    # Targets
    parser.add_argument(
        "--target-size",
        type=int,
        default=20,
        help="Number of target classes.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="model_reset",
        help="Name of the new checkpoint.",
    )
    # Dataset
    parser.add_argument(
        "--coco",
        action="store_true",
        help="For COCO models."
    )
    args = parser.parse_args()
    return args


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.
    """

    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        weight_name = param_name + (".weight" if is_weight else ".bias")
        pretrained_weight = ckpt["state_dict"][weight_name]
        prev_cls = pretrained_weight.size(0)
        if "fc_cls" in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
        if args.coco:
            for idx, c in enumerate(BASE_CLASSES):
                if "fc_cls" in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[
                        IDMAP[c] * 4 : (IDMAP[c] + 1) * 4
                    ] = pretrained_weight[idx * 4 : (idx + 1) * 4]
        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        if "fc_cls" in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        ckpt["state_dict"][weight_name] = new_weight

    surgery_loop(args, surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. Feature extractor weights are
    from the base detector. Only the final layer weights are combined.
    """

    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        if not is_weight and param_name + ".bias" not in ckpt["state_dict"]:
            return
        weight_name = param_name + (".weight" if is_weight else ".bias")
        pretrained_weight = ckpt["state_dict"][weight_name]
        prev_cls = pretrained_weight.size(0)
        if "fc_cls" in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
        else:
            new_weight = torch.zeros(tar_size)
        if args.coco:
            for idx, c in enumerate(BASE_CLASSES):
                if "fc_cls" in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[
                        IDMAP[c] * 4 : (IDMAP[c] + 1) * 4
                    ] = pretrained_weight[idx * 4 : (idx + 1) * 4]
        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        ckpt2_weight = ckpt2["state_dict"][weight_name]
        if args.coco:
            for i, c in enumerate(NOVEL_CLASSES):
                if "fc_cls" in param_name:
                    new_weight[IDMAP[c]] = ckpt2_weight[i]
                else:
                    new_weight[
                        IDMAP[c] * 4 : (IDMAP[c] + 1) * 4
                    ] = ckpt2_weight[i * 4 : (i + 1) * 4]
            if "fc_cls" in param_name:
                new_weight[-1] = pretrained_weight[-1]
        else:
            if "fc_cls" in param_name:
                new_weight[prev_cls:-1] = ckpt2_weight[:-1]
                new_weight[-1] = pretrained_weight[-1]
            else:
                new_weight[prev_cls:] = ckpt2_weight
        ckpt["state_dict"][weight_name] = new_weight

    surgery_loop(args, surgery)


def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.src1, map_location=torch.device("cpu"))
    if not args.keep_student_teacher:
        ckpt = extract_teacher(ckpt)
    if args.method == "combine":
        ckpt2 = torch.load(args.src2, map_location=torch.device("cpu"))
        if not args.keep_student_teacher:
            ckpt2 = extract_teacher(ckpt2)
        save_name = args.target_name + "_combine.pth"
    else:
        ckpt2 = None
        save_name = (
            args.target_name
            + "_"
            + ("remove" if args.method == "remove" else "surgery")
            + ".pth"
        )
    if args.save_dir == "":
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    # Remove parameters
    if args.method == "remove":
        for param_name in args.param_name:
            del ckpt["state_dict"][param_name + ".weight"]
            if param_name + ".bias" in ckpt["state_dict"]:
                del ckpt["state_dict"][param_name + ".bias"]
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
    for idx, (param_name, tar_size) in enumerate(
        zip(args.param_name, tar_sizes)
    ):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    # Save to file
    save_ckpt(ckpt, save_path)


def extract_teacher(ckpt):
    keys = list(ckpt["state_dict"].keys())
    for key in keys:
        if "student" in key:
            ckpt["state_dict"].pop(key)
            continue
        new_key = key.replace("teacher.", "")
        ckpt["state_dict"][new_key] = ckpt["state_dict"].pop(key)
    return ckpt

    
def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print("New ckpt saved to {}".format(save_name))


def reset_ckpt(ckpt):
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0


if __name__ == "__main__":
    args = parse_args()
    if args.keep_student_teacher:
        args.param_name = [
            "student.roi_head.bbox_head.fc_cls",
            "student.roi_head.bbox_head.fc_reg",
            "teacher.roi_head.bbox_head.fc_cls",
            "teacher.roi_head.bbox_head.fc_reg",
        ]
    else:
        args.param_name = [
            "roi_head.bbox_head.fc_cls",
            "roi_head.bbox_head.fc_reg",
        ]
    if args.coco:
        # COCO
        # fmt: off
        NOVEL_CLASSES = [
            1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67,
            72,
        ]
        BASE_CLASSES = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        # fmt: on
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
        TAR_SIZE = 80
    else:
        # VOC-like format
        TAR_SIZE = args.target_size

    if args.method == "combine":
        combine_ckpts(args)
    else:
        ckpt_surgery(args)
