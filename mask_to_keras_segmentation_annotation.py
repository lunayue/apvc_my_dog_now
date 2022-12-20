import os

import cv2
import numpy as np

TRAIN_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/train"
VAL_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/validation"

TRAIN_ANNOTATIONS = "C:/Users/susan/fiftyone/coco-2017/keras_segmentation_annotation/train"
VAL_ANNOTATIONS = "C:/Users/susan/fiftyone/coco-2017/keras_segmentation_annotation/validation"


#Guarda as imagens do file_path num dicionário com o nome do ficheiro
def fetch_images(file_path):
    result = {}
    for path in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, path)):
            img = cv2.imread(os.path.join(file_path, path), 1)
            result[path] = img
    return result


def create_annotation(mask_location, save_location):
    masks = fetch_images(mask_location)
    for file_name in masks:
        annotation = np.array(masks[file_name])
        annotation[annotation < 240] = 0 # vlue 240 achieved by trial and error
        annotation[annotation >= 240] = 1
        cv2.imwrite(save_location + "/" + file_name.split(".")[0] + ".png", annotation)


create_annotation(VAL_MASKS, VAL_ANNOTATIONS)
print("done validation pics")

create_annotation(TRAIN_MASKS, TRAIN_ANNOTATIONS)
print("done train pics")
