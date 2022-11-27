import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone as fo

# A name for the dataset
name = "my-dataset"

# The directory containing the dataset to import
dataset_dir = "C:/Users/susan/fiftyone/coco-2017/validation"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
    label_types=["segmentations"],
    classes=["dog"],
    only_matching=True
)
session = fo.launch_app(dataset)
session.wait()
