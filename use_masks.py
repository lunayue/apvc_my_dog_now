import os

import cv2

VAL_PICS = "C:/Users/susan/fiftyone/coco-2017/validation/data"
VAL_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/validation"
VAL_NOPUPS = "C:/Users/susan/fiftyone/coco-2017/no_pups/validation"

TRAIN_PICS = "C:/Users/susan/fiftyone/coco-2017/train/data"
TRAIN_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/train"
TRAIN_NOPUPS = "C:/Users/susan/fiftyone/coco-2017/no_pups/train"


#Guarda as imagens do file_path num dicionário com o nome do ficheiro
def fetch_images(file_path):
    result = {}
    for path in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, path)):
            img = cv2.imread(os.path.join(file_path, path), 1)
            result[path] = img
    return result


def remove_pups(original_location, mask_location, save_location):
    originals = fetch_images(original_location)
    masks = fetch_images(mask_location)
    for file_name in originals:
        original = originals[file_name]
        (mask, g, r) = cv2.split(masks[file_name])
        no_pups = cv2.bitwise_and(original, original, mask=mask)
        cv2.imwrite(save_location + "/" + file_name, no_pups)


remove_pups(VAL_PICS, VAL_MASKS, VAL_NOPUPS)
print("done validation pics")

remove_pups(TRAIN_PICS, TRAIN_MASKS, TRAIN_NOPUPS)
print("done train pics")