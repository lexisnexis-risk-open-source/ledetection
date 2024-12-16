from lvis import LVIS as BaseLVIS


class LVIS(BaseLVIS):
    def get_cat_ids(self, cat_names=None):
        if cat_names:
            ids = [cat["id"] for cat in self.dataset["categories"] if cat["name"] in cat_names]
        else:
            ids = list(self.cats.keys())
        return ids
