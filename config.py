import os

APP_ROOT = os.path.dirname(__file__)
#

train_set_dir = f"{APP_ROOT}/data/images"
#test_set_dir = "/Volumes/SSD1T/t2/images/test_set"

#train_set_dir = "../data/mini_imagenet100/images/train_set"
#test_set_dir = "../data/mini_imagenet100/images/test_set"


image_extension = ".png"

#image_tags = "/Volumes/SSD1T/t2/tags"
image_classes_file = f"{APP_ROOT}/data/image_classes.txt"
num_classes = 9176


image_resize_width = 512
image_resize_height = 512

batch_size = 10

transform_normalization_file = "output/image_transform_normalization.txt"
transform_normalization_mean = [0.5894, 1.3996, 1.7191]
transform_normalization_std = [0.2528, 0.2523, 0.2520]



