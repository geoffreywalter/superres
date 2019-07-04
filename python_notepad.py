import random
import glob
from PIL import Image
import numpy as np

# def perceptual_distance_np(y_true, y_pred):
    # """Calculate perceptual distance, DO NOT ALTER"""
    # rmean = (y_true[ :, :, 0] + y_pred[ :, :, 0]) / 2
    # r = y_true[ :, :, 0] - y_pred[ :, :, 0]
    # g = y_true[ :, :, 1] - y_pred[ :, :, 1]
    # b = y_true[ :, :, 2] - y_pred[ :, :, 2]
    # return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

# zero = np.zeros((256, 256, 3))
# ones = np.ones((256, 256, 3))

# input_images = np.zeros((256, 256, 3))
# img = np.asarray(Image.open("8380716-rose-out.jpg"))

# grey_img = np.zeros((256, 256, 3))
# grey_img_grey = (Image.open("8380716-rose-out.jpg").convert('L')).convert('RGB')
# grey_img_grey.save("grey_img_grey.png")
# grey_img_rgb = grey_img_grey.convert('RGB')
# grey_img_rgb.save("grey_img_rgb.png")
# grey_img = ones - np.asarray(grey_img_rgb)


# print("perceptual_distance " + str(perceptual_distance_np(img, grey_img)))


img = np.zeros((5*3*256, 256, 3))
img = np.asarray(Image.open("srdensenet_sample_test.jpg"))
img_reshaped = []
img_reshaped = np.split(img, 5, axis=1)
img_out = Image.fromarray(np.concatenate(img_reshaped, axis=0))

img_out.save("srdensenet_sample_test_reshaped.jpg")
