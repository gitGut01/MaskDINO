import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

def load_coco_with_rle(json_file, image_root):
    """
    Load a COCO-style dataset with RLE masks.
    Args:
        json_file (str): Path to the COCO-style JSON annotations.
        image_root (str): Path to the directory containing images.
    Returns:
        list[dict]: A list of dictionaries in Detectron2's dataset format.
    """
    coco = COCO(json_file)
    dataset_dicts = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        record = {
            "file_name": os.path.join(image_root, img_info["file_name"]),
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
            "annotations": []
        }

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            # Decode RLE if present
            if isinstance(ann["segmentation"], dict):  # RLE
                mask = mask_utils.decode(ann["segmentation"])
            else:  # Polygon
                mask = None

            record["annotations"].append({
                "bbox": ann["bbox"],
                "bbox_mode": 1,  # BoxMode.XYWH_ABS
                "category_id": ann["category_id"],
                "segmentation": ann["segmentation"],  # Keep RLE/Polygon
                "iscrowd": ann["iscrowd"]
            })
        dataset_dicts.append(record)

    return dataset_dicts


def register_coco_rle_dataset(name, json_file, image_root, metadata=None):
    """
    Register a COCO-style dataset with RLE masks.
    Args:
        name (str): The name of the dataset.
        json_file (str): Path to the COCO-style JSON annotations.
        image_root (str): Path to the directory containing images.
        metadata (dict): Optional metadata for the dataset.
    """
    DatasetCatalog.register(name, lambda: load_coco_with_rle(json_file, image_root))
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type="coco", **(metadata or {}))


# Example usage
register_coco_rle_dataset(
    name="my_rle_dataset_train",
    json_file="test/coco_annot.json",
    image_root="test/images",
    metadata={"thing_classes": ["class1"]}
)

# register_coco_rle_dataset(
#     name="my_coco_dataset_val",
#     json_file="path/to/annotations_val.json",
#     image_root="path/to/val/images",
#     metadata={"thing_classes": ["class1", "class2", "class3"]}
# )
