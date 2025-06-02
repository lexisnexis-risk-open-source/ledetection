import collections
import copy

import numpy as np
from mmcv.utils import build_from_cfg

from mmdet.datasets import (
    DATASETS,
    PIPELINES,
    ConcatDataset,
)
from ledet.datasets import build_dataset


@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semi-supervised detection."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]


@DATASETS.register_module()
class MultiSourceImageMixDataset:

    def __init__(
        self,
        datasets,
        pipeline,
        skip_type_keys=None,
        max_refetch=15,
    ):
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform["type"])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError("pipeline must be a dict")

        self.max_refetch = max_refetch

        self.datasets = datasets
        self.dataset_sizes = [len(dset) for dset in datasets]
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], "flag"):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

        self.max_num_pasted = 100

    def select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results["gt_bboxes"]
        labels = results["gt_labels"]
        masks = results["gt_masks"]
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results["gt_bboxes"] = selected_bboxes
        results["gt_labels"] = selected_labels
        results["gt_masks"] = selected_masks
        return results

    def _resize_bboxes(self, bboxes, scale_factor, img_shape, bbox_clip_border=True):
        """Resize bounding boxes with `scale_factor`."""
        bboxes = bboxes * scale_factor
        if bbox_clip_border:
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        return bboxes

    def copy_paste(self, results):
        """CopyPaste from labeled image to unlabeled image."""
        sup_res = results[0]
        sup_res = self.select_object(sup_res)
        unsup_res = results[1]
        sup_shape = sup_res["img_shape"]
        unsup_shape = unsup_res["img_shape"]
        max_h, max_w = [max(sup_shape[i], unsup_shape[i]) for i in range(2)]
        mask_canvas = np.zeros(shape=(max_h, max_w), dtype=np.uint8)
        src_canvas = 128 * np.ones(shape=(max_h, max_w, 3), dtype=np.uint8)
        dst_canvas = src_canvas.copy()
        src_img = sup_res["img"]
        dst_img = unsup_res["img"]
        src_canvas[:sup_shape[0], :sup_shape[1], :] = src_img
        src_img = src_canvas
        dst_canvas[:unsup_shape[0], :unsup_shape[1], :] = dst_img
        dst_img = dst_canvas

        src_masks = np.where(np.any(sup_res["gt_masks"].masks, axis=0), 1, 0)
        src_masks = src_masks.astype(np.uint8)
        mask_canvas[:sup_shape[0], :sup_shape[1]] = src_masks
        src_masks = mask_canvas[..., None]

        dst_img = dst_img * (1 - src_masks) + src_img * src_masks
        unsup_res["img"] = dst_img
        unsup_res["bbox_fields"] = ["gt_bboxes"]
        unsup_res["gt_labels"] = sup_res["gt_labels"]
        unsup_res["gt_bboxes"] = sup_res["gt_bboxes"]
        unsup_res["img_shape"] = sup_res["img_shape"] = (max_h, max_w, 3)

        return unsup_res

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, idx):
        sup_idx = idx % self.dataset_sizes[0]
        unsup_idx = idx % self.dataset_sizes[1]

        sup_results = self.datasets[0][sup_idx]
        unsup_results = self.datasets[1][unsup_idx]

        mix_results = [copy.deepcopy(res) for res in [sup_results, unsup_results]]
        results = self.copy_paste(mix_results)

        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue
            for _ in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    "The training pipeline of the dataset wrapper"
                    " always return None. Please check the correctness "
                    "of the dataset and its pipeline.")

        return results
