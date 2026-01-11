import os

APP_ROOT = os.path.dirname(__file__)
#

train_set_dir = f"{APP_ROOT}/data/images"
#test_set_dir = "/Volumes/SSD1T/t2/images/test_set"

num_classes = 9176
all_tags = f"{APP_ROOT}/data/tags.txt"
safe_tags = f"{APP_ROOT}/data/safe_tags.txt"

image_extension = ".png"


current_tags = f"{APP_ROOT}/output/DeepDanbooruModel/current_tags.txt"
current_safe_tags = f"{APP_ROOT}/output/DeepDanbooruModel/current_safe_tags.txt"
current_without_tags = f"{APP_ROOT}/output/DeepDanbooruModel/current_without_tags.txt"

image_resize_width = 512
image_resize_height = 512

batch_size = 10

transform_normalization_file = "output/image_transform_normalization.txt"
transform_normalization_mean = [0.5894, 1.3996, 1.7191]
transform_normalization_std = [0.2528, 0.2523, 0.2520]



