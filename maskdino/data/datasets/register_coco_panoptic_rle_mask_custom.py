import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from pycocotools import mask as mask_utils


def load_rle_dataset(json_file, image_root):
    """
    Load a dataset with RLE masks into Detectron2's standard format.
    Args:
        json_file (str): Path to the JSON file containing annotations.
        image_root (str): Path to the directory containing images.
    Returns:
        list[dict]: A list of dictionaries in Detectron2's dataset format.
    """
    with open(json_file) as f:
        annotations = json.load(f)
    
    dataset_dicts = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in dataset_dicts:
            dataset_dicts[image_id] = {
                "file_name": os.path.join(image_root, image_id),
                "image_id": image_id,
                "height": ann["segmentation"]["size"][0],
                "width": ann["segmentation"]["size"][1],
                "annotations": [],
            }

        rle = ann["segmentation"]
        dataset_dicts[image_id]["annotations"].append({
            "bbox": ann["bbox"],
            "bbox_mode": 1,  # BoxMode.XYWH_ABS
            "category_id": ann["category_id"],
            "segmentation": rle,  # Keep the RLE directly
            "area": ann["area"],
            "iscrowd": ann["iscrowd"],
        })

    return list(dataset_dicts.values())


def register_rle_dataset(name, json_file, image_root, metadata=None):
    """
    Register a custom dataset with RLE masks.
    Args:
        name (str): The name of the dataset.
        json_file (str): Path to the JSON file containing annotations.
        image_root (str): Path to the directory containing images.
        metadata (dict): Optional metadata for the dataset.
    """
    DatasetCatalog.register(name, lambda: load_rle_dataset(json_file, image_root))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",  # Use COCO evaluator for COCO-style datasets
        **(metadata or {})
    )


# Example usage
register_rle_dataset(
    name="my_rle_dataset_train",
    json_file="test/coco_annot.json",
    image_root="test/images",
    metadata={"thing_classes": ["class1"]}
)

# register_rle_dataset(
#     name="my_rle_dataset_val",
#     json_file="path/to/annotations_val.json",
#     image_root="path/to/val/images",
#     metadata={"thing_classes": ["class1", "class2", "class3"]}
# )
