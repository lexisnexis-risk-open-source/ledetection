from mmdet.models import HEADS
from mmdet.models.dense_heads import RPNHead as BaseRPNHead

from ledet.utils.structure_utils import freeze_module


@HEADS.register_module(force=True)
class RPNHead(BaseRPNHead):
    def __init__(self, frozen=False, *args, **kwargs):
        super(RPNHead, self).__init__(*args, **kwargs)
        freeze_module(self, freeze=frozen)