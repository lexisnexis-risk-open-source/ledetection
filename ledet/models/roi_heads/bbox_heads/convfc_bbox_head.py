from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead as BaseConvFCBBoxHead

from ledet.utils.structure_utils import freeze_module


@HEADS.register_module(force=True)
class ConvFCBBoxHead(BaseConvFCBBoxHead):
    def __init__(self, frozen_layer_names=None, *args, **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        if frozen_layer_names:
            layer_names = frozen_layer_names.split() \
                if not isinstance(frozen_layer_names, (tuple, list)) \
                else frozen_layer_names
            for layer in layer_names:
                module = getattr(self, layer)
                freeze_module(module, True)


@HEADS.register_module(force=True)
class Shared2FCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fc_out_channels=1024,
                 frozen_layer_names=None,
                 *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            frozen_layer_names=frozen_layer_names,
            *args,
            **kwargs)


@HEADS.register_module(force=True)
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fc_out_channels=1024,
                 frozen_layer_names=None,
                 *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            frozen_layer_names=frozen_layer_names,
            *args,
            **kwargs)
