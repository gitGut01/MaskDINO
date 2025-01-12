from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_coco_rle_mask_dataset", {}, "test/coco_annot.json", "test/images")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")