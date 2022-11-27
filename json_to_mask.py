import json
import os

import cv2
import numpy as np

from PIL import Image, ImageDraw

# consulted bibliography:
# https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

TRAIN_JSON = "C:/Users/susan/fiftyone/coco-2017/train/labels.json"
VAL_JSON = "C:/Users/susan/fiftyone/coco-2017/validation/labels.json"
TEST_JASON = "C:/Users/susan/fiftyone/coco-2017/test/labels.json"

STORE_VAL_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/validation"

val_dic = json.load(open(VAL_JSON))
#for aux in val_dic["annotations"]:
#    print("segmentation: " + str(aux["segmentation"]))
#    print("image id: " + str(aux["image_id"]))
#    print("category id: " + str(aux["category_id"]))


def create_mask_files(labels_location, masks_location):
    read_json = json.load(open(labels_location))
    for image in read_json["images"]:
        width = image["width"]
        height = image["height"]
        mask_name = image["file_name"]
        if not os.path.isfile(masks_location + "/" + mask_name):
            empty_mask = np.ones((height, width), np.uint8) * 255
            cv2.imwrite(masks_location + "/" + mask_name, empty_mask)


def make_masks(labels_location, masks_location, category_id):
    read_json = json.load(open(labels_location))
    for annotation in read_json["annotations"]:
        if annotation["category_id"] == category_id:
            mask_image_id = annotation["image_id"]
            if len(str(mask_image_id)) < 12:
                mask_image_id = "0" * (12-len(str(mask_image_id))) + str(mask_image_id)
            mask_file_name = masks_location + "/" + mask_image_id + ".jpg"
            mask_image = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
            height, width = mask_image.shape
            if annotation["iscrowd"] == 0:
                polygons = []
                for segment in annotation["segmentation"]:
                    for dot in range(0,len(segment),2):
                        x = segment[dot]
                        y = segment[dot+1]
                        polygons.append((int(x), int(y)))
                    img = Image.new('L', (width, height), 255)
                    ImageDraw.Draw(img).polygon(polygons, outline=1, fill=0)
                    new_mask = np.array(img)
                    full_mask = cv2.bitwise_and(mask_image, new_mask)
                    cv2.imwrite(mask_file_name, full_mask)

            elif annotation["iscrowd"] == 1:
                counts = annotation["segmentation"]["counts"]
                new_mask = np.ones(width*height, np.uint8)*255
                painted_pixels = 0
                for count in range(1, len(counts), 2):
                    painted_pixels += counts[count-1]
                    new_mask[painted_pixels: painted_pixels+counts[count]] = 0
                    painted_pixels += counts[count]

                new_mask = np.reshape(new_mask, (width, height))
                new_mask = new_mask.T
                full_mask = cv2.bitwise_and(mask_image, new_mask)
                cv2.imwrite(mask_file_name, full_mask)



create_mask_files(VAL_JSON, STORE_VAL_MASKS)
print("empty masks done")
make_masks(VAL_JSON, STORE_VAL_MASKS, 18)
print("masks with pups done")
