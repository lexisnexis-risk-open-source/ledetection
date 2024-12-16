from mmdet.datasets import DATASETS
from mmdet.datasets import LVISV1Dataset as BaseLVISV1Dataset
from .api_wrappers import LVIS


@DATASETS.register_module(force=True)
class LVISV1Dataset(BaseLVISV1Dataset):
    def load_annotations(self, ann_file):
        self.coco = LVIS(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # Extract the ``filename`` from ``coco_url``.
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            info["filename"] = info["coco_url"].replace(
                "http://images.cocodataset.org/", "")
            data_infos.append(info)
        return data_infos

