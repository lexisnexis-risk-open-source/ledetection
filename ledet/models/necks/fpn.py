from mmdet.models import NECKS
from mmdet.models.necks import FPN as BaseFPN

from ledet.utils.structure_utils import freeze_module


@NECKS.register_module(force=True)
class FPN(BaseFPN):
    def __init__(self, frozen=False, *args, **kwargs):
        super(FPN, self).__init__(*args, **kwargs)
        freeze_module(self, freeze=frozen)