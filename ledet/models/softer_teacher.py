import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi
from mmdet.models import DETECTORS, build_loss

from ledet.utils import log_image_with_boxes
from ledet.utils.structure_utils import weighted_loss

from .soft_teacher import SoftTeacher
from .utils import Transform2D


@DETECTORS.register_module()
class SoftERTeacher(SoftTeacher):
    def __init__(self, *args, **kwargs):
        super(SoftERTeacher, self).__init__(*args, **kwargs)
        if self.train_cfg is not None:
            self.unsup_weight_beta = self.train_cfg.get(
                "unsup_weight_beta",
                2.0 * self.unsup_weight_alpha
            )
            if not any([loss_type == self.train_cfg.sim_cls_loss.type
                        for loss_type in [
                            "CrossEntropySimilarityLoss",
                            "KnowledgeDistillationKLDivLoss",
                            "CrossEntropyLoss"
                        ]]):
                raise NotImplementedError(
                    "Only `CrossEntropySimilarityLoss`, `KnowledgeDistillationKLDivLoss`, "
                    "or `CrossEntropyLoss` is currently supported for proposal bbox similarity."
                )
            if "IoULoss" not in self.train_cfg.iou_bbox_loss.type:
                raise NotImplementedError(
                    "A type of `IoULoss` is needed for proposal bbox regression. "
                    "Only standard `IoULoss` has been tested."
                )
            self.sim_cls_loss = build_loss(self.train_cfg.sim_cls_loss)
            self.iou_bbox_loss = build_loss(self.train_cfg.iou_bbox_loss)

    def forward_train(self, img, img_metas, **kwargs):
        loss = super().forward_train(img, img_metas, **kwargs)
        unsup_loss_weight = self.loss_weight * self.unsup_weight_beta
        aux_loss = weighted_loss(
            self.compute_aux_loss(self.student_info, self.teacher_info),
            unsup_loss_weight,
        )
        aux_loss = {"unsup_" + k: v for k, v in aux_loss.items()}
        loss.update(**aux_loss)
        return loss
    
    def compute_aux_loss(
        self,
        student_info,
        teacher_info,
        **kwargs,
    ):
        teacher_proposals = teacher_info["candidate_proposals"]
        student_proposals = student_info["candidate_proposals"]
        
        teacher_rois = bbox2roi(teacher_proposals)
        student_rois = bbox2roi(student_proposals)
        
        bbox_results = self.student.roi_head._bbox_forward(student_info["backbone_feature"], student_rois)
        bbox_targets = self.teacher.roi_head._bbox_forward(teacher_info["backbone_feature"], teacher_rois)
        
        with torch.no_grad():
            pred_labels = torch.argmax(bbox_targets['cls_score'], dim=-1)
            fg_inds = (pred_labels >= 0) & (pred_labels < self.num_classes)
            
            bbox_target = bbox_targets["bbox_pred"]
            bbox_target = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                teacher_rois[:, 1:], bbox_target
            )
            bbox_target = torch.cat([teacher_rois[:, :1], bbox_target], dim=-1)
            bbox_target_list = self._roi2bbox(bbox_target, self.num_classes)
            # Transform teacher bboxes to student space using student matrix
            M_student = student_info["student_matrix"]
            bbox_target_list = self._transform_bbox_class_aware(
                bbox_target_list,
                M_student,
                [meta["img_shape"] for meta in student_info["img_metas"]],
                self.num_classes
            )
            bbox_target = torch.cat(bbox_target_list, dim=0)
            pos_bbox_target = bbox_target[fg_inds.type(torch.bool), pred_labels[fg_inds.type(torch.bool)]]
            
        bbox_pred = bbox_results["bbox_pred"].clone()
        bbox_pred = self.student.roi_head.bbox_head.bbox_coder.decode(
            student_rois[:, 1:], bbox_pred
        )
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_bbox_pred = bbox_pred[fg_inds.type(torch.bool), pred_labels[fg_inds.type(torch.bool)]]

        if self.debug_mode:
            teacher_props = teacher_info["proposals"]
            teacher_props = [p[p[:, 4] > 0.99] for p in teacher_props]
            if len(teacher_props[0]) > 0:
                student_props = self._transform_bbox(
                    teacher_props,
                    M_student,
                    [meta["img_shape"] for meta in student_info["img_metas"]],
                )
                log_image_with_boxes(
                    interval=500,
                    tag="student_rpn",
                    image=student_info["img"][0],
                    bboxes=student_props[0][:, :4],
                    bbox_tag="student_proposals",
                    img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
                    filename=student_info["img_metas"][0]["filename"].split("/")[-1],
                )
                log_image_with_boxes(
                    interval=500,
                    tag="teacher_rpn",
                    image=teacher_info["img"][0],
                    bboxes=teacher_props[0][:, :4],
                    bbox_tag="teacher_proposals",
                    img_norm_cfg=teacher_info["img_metas"][0]["img_norm_cfg"],
                    filename=teacher_info["img_metas"][0]["filename"].split("/")[-1],
                )

        return self.aux_loss(
            bbox_results["cls_score"],
            bbox_targets["cls_score"],
            pos_bbox_pred,
            pos_bbox_target,
            fg_inds,
        )

    def aux_loss(
        self,
        cls_scores,
        cls_labels,
        bbox_preds,
        bbox_targets,
        label_weights,
    ):
        loss = dict()
        bbox_avg_factor = max(cls_scores.size(0), 1.0)
        cls_avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
        
        sim_cls_loss = torch.tensor([0.0]).to(cls_scores.device)
        if cls_scores is not None:
            if cls_scores.numel() > 0:
                sim_cls_loss = self.sim_cls_loss(
                    cls_scores,
                    cls_labels,
                    weight=label_weights,
                    avg_factor=cls_avg_factor
                )
        loss["loss_sim_cls"] = sim_cls_loss
        
        iou_bbox_loss = torch.tensor([0.0]).to(bbox_preds.device)
        if bbox_preds is not None:
            if bbox_preds.numel() > 0:
                iou_bbox_loss = self.iou_bbox_loss(
                    bbox_preds,
                    bbox_targets,
                    avg_factor=bbox_avg_factor
                )
        loss["loss_iou_bbox"] = iou_bbox_loss
        return loss
    
    def _roi2bbox(self, rois, num_classes):
        """Convert rois to bounding box format.

        Args:
            rois (torch.Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            num_classes (int): Number of classes for the dataset.

        Returns:
            list[torch.Tensor]: Converted boxes of corresponding rois and classes.
        """
        bbox_list = []
        img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
        for img_id in img_ids:
            inds = (rois[:, 0] == img_id.item())
            bbox = rois[inds, 1:].view(-1, num_classes, 4)
            bbox_list.append(bbox)
        return bbox_list
    
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox_class_aware(
        self, bboxes, trans_mat, max_shape, num_classes
    ):
        bboxes = Transform2D.transform_bboxes(
            bboxes, trans_mat, max_shape, num_classes
        )
        return bboxes
