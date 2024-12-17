import cv2
import mmcv
import numpy as np

from mmdet.core import BitmapMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.transforms import CopyPaste as BaseCopyPaste


@PIPELINES.register_module(force=True)
class CopyPaste(BaseCopyPaste):
    
    def __init__(
        self,
        backend="cv2",
        interpolation="bicubic",
        bbox_clip_border=True,
        **kwargs
    ):
        self.backend = backend
        self.interpolation = interpolation
        self.bbox_clip_border = bbox_clip_border
        super(CopyPaste, self).__init__(**kwargs)
    
    def _resize_bboxes(self, bboxes, scale_factor, img_shape):
        """Resize bounding boxes with `scale_factor`."""
        bboxes = bboxes * scale_factor
        if self.bbox_clip_border:
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        return bboxes

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results["img"]
        dst_img_shape = dst_img.shape
        dst_bboxes = dst_results["gt_bboxes"]
        dst_labels = dst_results["gt_labels"]
        dst_masks = dst_results["gt_masks"]
        
        src_img = src_results["img"]
        src_img_shape = src_img.shape
        src_bboxes = src_results["gt_bboxes"]
        src_labels = src_results["gt_labels"]
        src_masks = src_results["gt_masks"]
        
        if len(src_bboxes) == 0:
            if self.paste_by_box:
                dst_results.pop("gt_masks")
            return dst_results
        
        if src_img_shape != dst_img_shape:
            # Resize `src_img`, `src_bboxes`, and `src_masks` to match `dst_results`.
            h, w = dst_img_shape[:2]
            src_img, w_scale, h_scale = mmcv.imresize(
                src_img,
                size=(w, h),
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend
            )
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale],
                dtype=np.float32
            )
            src_bboxes = self._resize_bboxes(
                src_bboxes,
                scale_factor,
                img_shape=(h, w),
            )
            src_masks = src_masks.resize((h, w))
        
        # Update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        composed_mask = composed_mask.astype(np.uint8)
        
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes()
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # Filter totally occluded objects
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])

        dst_results["img"] = img
        dst_results["gt_bboxes"] = bboxes
        dst_results["gt_labels"] = labels
        if self.paste_by_box:
            dst_results.pop("gt_masks")
        else:
            dst_results["gt_masks"] = BitmapMasks(
                masks,
                masks.shape[1],
                masks.shape[2]
            )

        return dst_results
