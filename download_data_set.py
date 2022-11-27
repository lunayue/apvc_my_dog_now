import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    #split="validation",
    label_types=["segmentations"],
    #classes=["dog"],
    #only_matching=True
)

session = fo.launch_app(dataset)
session.wait()
