from detectron2.data import detection_utils as utils
from detectron2.utils import MaskFormat


def CocoRleDatasetMapper(dataset_dict):
    """
    A custom mapper function to handle RLE mask formatting correctly.
    """
    annotations = dataset_dict["annotations"]
    
    # Use RLE format to mask the annotation properly
    instances = utils.annotations_to_instances(
        annotations, 
        image_shape=(dataset_dict["height"], dataset_dict["width"]), 
        mask_format=MaskFormat.RLE
    )
    
    dataset_dict["instances"] = instances
    return dataset_dict