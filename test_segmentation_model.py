import os
import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model
import tensorflow as tf

TEST_PICS = "C:/Users/susan/fiftyone/coco-2017/test/data"
TEST_MASKS = "C:/Users/susan/fiftyone/coco-2017/masks/test"

MODEL_LOCATION = "pspnet_50_ft_1"


def fetch_images(file_path):
    result = {}
    for path in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, path)):
            img = cv2.imread(os.path.join(file_path, path), 1)
            result[path] = img
    return result


def use_model_on_pic(model, pic):
    # Prepping the pic to be handed to the model
    resized = cv2.resize(pic, (473, 473))
    as_array = np.array(resized)
    as_array = as_array.reshape((1, 473, 473, 3))
    as_tensor = tf.convert_to_tensor(as_array)

    # Using the model
    result_tensor = model.predict(as_tensor)

    # Turning the result into a mask
    pixel_class = []
    # É possivel que dê para fazer isto com menos código
    # Ou de forma mais eficiente
    for p in result_tensor[0]:
        if p[0] > p[1]: # is pup
            pixel_class.append(0)
        else: # isn't pup
            pixel_class.append(255)

    height, width, channels = pic.shape
    mask = np.array(pixel_class, np.uint8)
    mask = mask.reshape((473, 473))
    mask = cv2.resize(mask, (width, height))
    return mask


def create_masks(original_location, model_location, save_location):
    model = load_model(model_location)
    originals = fetch_images(original_location)
    for file_name in originals:
        mask = use_model_on_pic(model, originals[file_name])
        cv2.imwrite(save_location + "/" + model_location + "/" + file_name, mask)


create_masks(TEST_PICS, MODEL_LOCATION, TEST_MASKS)
