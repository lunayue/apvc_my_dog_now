import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model
import tensorflow as tf

# ESTE SERVIU PARA FAZER EXPERIENCIAS

model = load_model("pspnet_50_ft_1")

test_img_input = cv2.imread("C:\\Users\\susan\\fiftyone\\coco-2017\\test\\data\\Female-Dogs-in-Heat-IE-768x512.jpg")

height, width, channels = test_img_input.shape

test2 = cv2.resize(test_img_input, (473, 473))

array = np.array(test2)

array = array.reshape((1, 473, 473, 3))

tensor = tf.convert_to_tensor(array)

aux = model.predict(tensor)
print(aux.shape)
#print(aux)
print(aux[0][0:2])
#print(aux[0][0])

#if aux[0][0][0] > aux[0][0][1]:
#    print("dog")
#else:
#    print("not a dog")

pixel_class = []
for p in aux[0]:
    #print(p)
    if p[0] > p[1]:
        pixel_class.append(0)
    else:
        pixel_class.append(255)

dark_mask = np.array(pixel_class, np.uint8)
dark_mask = dark_mask.reshape((473, 473))
dark_mask = cv2.resize(dark_mask, (width, height))
cv2.imshow("mask?", dark_mask)
cv2.imshow("original", test_img_input)
cv2.waitKey(0)
