import torch
import torch.nn as nn

from mmdet.models import LOSSES
from mmdet.models.losses import weighted_loss


@weighted_loss
def cross_entropy_similarity_loss(pred,
                                  target,
                                  detach_target=True,
                                  target_probs=False):
    r"""Loss function for cross entropy similarity.
    Args:
        pred (Tensor): Predicted logits with shape (N, C), C is the number
            of classes.
        target (Tensor): Target logits or probabilities with shape (N, C).
        detach_target (bool): Remove target from automatic differentiation.
        target_probs (bool): Whether target is already in softmax probabilities.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == target.size()
    pred = pred.float()
    target = target.float()
    if not target_probs:
        target = torch.softmax(target, dim=1)
    if detach_target:
        target = target.detach()
    ces_loss = (-torch.log_softmax(pred, dim=1) * target).mean(1)
    return ces_loss


@LOSSES.register_module()
class CrossEntropySimilarityLoss(nn.Module):
    """Loss function for cross entropy similarity.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Weight multiplier of current loss.
        target_probs (bool): Whether target is already in softmax probabilities.
    """

    def __init__(self,
                 reduction="mean",
                 loss_weight=1.0,
                 target_probs=False):
        super(CrossEntropySimilarityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.target_probs = target_probs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (Tensor): Predicted logits with shape (N, C), C is the number
                of classes.
            target (Tensor): Target logits or probabilities with shape (N, C).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction
        )
        loss_ces = self.loss_weight * cross_entropy_similarity_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            target_probs=self.target_probs
        )
        return loss_ces
