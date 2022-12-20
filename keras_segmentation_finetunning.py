from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50
from tensorflow.python.keras.saving.save import load_model
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

TRAIN_PICS = "C:\\Users\\susan\\fiftyone\\coco-2017\\train\\data"
TRAIN_ANNOTATIONS = "C:\\Users\\susan\\fiftyone\\coco-2017\\keras_segmentation_annotation\\train"
VAL_PICS = "C:\\Users\\susan\\fiftyone\\coco-2017\\validation\\data"
VAL_ANNOTATIONS = "C:\\Users\\susan\\fiftyone\\coco-2017\\keras_segmentation_annotation\\validation"


def finetune_pspnet_50(save_location):
    pretrained_model = pspnet_50_ADE_20K()

    new_model = pspnet_50(n_classes=2)

    # transfer weights from pre-trained model to your model
    transfer_weights(pretrained_model, new_model)
    new_model.train(
        train_images=TRAIN_PICS,
        train_annotations=TRAIN_ANNOTATIONS,
        validate=True,
        val_images=VAL_PICS,
        val_annotations=VAL_ANNOTATIONS,
        epochs=10
    )

    new_model.save(save_location)
    print("saved" + save_location)


def finetune_finetuned(previous_model_location, new_model_location):
    pretrained_model = load_model(previous_model_location)

    new_model = pspnet_50(n_classes=2)

    # transfer weights from pre-trained model to your model
    transfer_weights(pretrained_model, new_model)
    new_model.train(
        train_images=TRAIN_PICS,
        train_annotations=TRAIN_ANNOTATIONS,
        validate=True,
        val_images=VAL_PICS,
        val_annotations=VAL_ANNOTATIONS,
        epochs=10
    )

    new_model.save(new_model_location)
    print("saved" + new_model_location)


finetune_pspnet_50("pspnet_50_ft_10")
finetune_finetuned("pspnet_50_ft_10", "pspnet_50_ft_20")

#out = new_model.predict_segmentation(
#    inp = "C:\\Users\\susan\\fiftyone\\coco-2017\\test\\data\\Female-Dogs-in-Heat-IE-768x512.jpg",
#    out_fname = "C:\\Users\\susan\\fiftyone\\coco-2017\\finetuningmodel\\Female-Dogs-in-Heat-IE-768x512.png"
#)

#test_img_input = cv2.imread("C:\\Users\\susan\\fiftyone\\coco-2017\\test\\data\\Female-Dogs-in-Heat-IE-768x512.jpg")

#test2 = cv2.resize(test_img_input, (473, 473))

#array = np.array(test2)

#array = array.reshape((1, 473, 473, 3))

#tensor = tf.convert_to_tensor(array)

#new_model.predict(tensor)

#cv2.imshow("pls work", out)
#cv2.waitKey(0)

