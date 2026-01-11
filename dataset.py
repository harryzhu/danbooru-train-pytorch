#!/usr/bin/env python
# coding: utf-8

import os
import io
import time
import glob
import hashlib
import random
import numpy as np
from PIL import Image

import skimage.transform

import torch
from torch.utils import data
from torch.utils.data import ConcatDataset

from torchvision import transforms
from torchvision.io import decode_image, decode_png
import matplotlib.pyplot as plt

import config as CFG 
from functions import *

transform = transforms.Compose([
	transforms.CenterCrop((CFG.image_resize_width,CFG.image_resize_height)),
	])

class mlxDeepDanbooruDataset(data.Dataset):
	def __init__(self, images, labels, transform, tags):
		self.images = images
		self.labels = labels
		self.tag_all_array = np.array(tags)
		self.transforms = transform


	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]

		ftemp = image
		t0 = Image.open(image)
		if t0.width != CFG.image_resize_width and t0.height != CFG.image_resize_height:		
			ftemp = f'{image.replace("/data/","/temp/")}'
			ftempdir = os.path.dirname(ftemp)
			if not os.path.exists(ftempdir):
				os.makedirs(ftempdir)
			if not os.path.exists(ftemp):
				pil_image = Image.open(image)
				pil_image = self.transforms(pil_image)
				pil_bytes = io.BytesIO()
				pil_image.save(pil_bytes, format='png') 
				# pil_image = pil_image.convert('RGB')
				pil_new = Image.open(pil_bytes)
				pil_new = pil_new.convert('RGB')
				pil_new.save(ftemp)
		try:
			pil_hwc = Image.open(ftemp)	
			#pil_hwc = pil_hwc.convert('RGB')
			pil_hwc = np.array(pil_hwc)
			#pil_chw = np.transpose(pil_hwc,(2,0,1))
			pil_chw = transform_and_pad_image(pil_hwc, CFG.image_resize_width,CFG.image_resize_height)

			image_data = pil_chw / 255.0

			tag_array = np.array(label)
			image_label = np.where(np.isin(self.tag_all_array, tag_array), 1, 0).astype(
				np.float32
			)
			#print(image_label, image_label.shape)
			#print(f'---read from dataset')
			return image_data, image_label
		except Exception as err:
			print(f'ERROR: {ftemp}')
			print(err)

	def __len__(self):
		return len(self.images)






def load_all_classes():
    c = []
    with open(f'{CFG.all_tags}','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "double_\\m/":
                line = "double_m"
            if len(line) > 0:
                c.append(line)
    return c

def load_image_classes2():
	c = []
	with open(CFG.image_classes_current,'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if len(line) > 0:
				c.append(line)
	return c


def images2labels(class_list):
	global all_classes
	safe_tags = []
	with open(f'{CFG.safe_tags}','r') as fr:
		lines = fr.readlines()
		for line in lines:
			line = line.strip()
			if len(line) > 0:
				safe_tags.append(line)



	all_train_images = []
	all_train_labels = []

	current_tags = []
	current_safe_tags = []
	#
	tag_files = glob.glob(f'{CFG.train_set_dir}/**/*.txt')
	#tag_files = random.shuffle(tag_files)
	for tag_file in tag_files:
		if tag_file[-5:-4] in ["0","6","c"]:
			continue
		png_file = ToUnixSlash(tag_file.replace(".txt",'.png'))
		if not os.path.exists(png_file):
			continue
		with open(tag_file,'r')as fr:
			fcontent = fr.read().strip(",").strip()
			words = fcontent.split(",")
			wsafe = []
			for word in words:
				word = word.strip()
				if len(word) == 0:
					continue
				if word == "double_\\m/":
					word = "double_m"
				current_tags.append(word)
				if word in safe_tags:
					wsafe.append(word)
					current_safe_tags.append(word)
			if len(wsafe) > 7:
				#print(wsafe)	
				#all_labels.append(words)
				all_train_labels.append(wsafe)
				all_train_images.append(png_file)

	current_tags = sorted(list(set(current_tags)))
	if current_tags:
		with open(CFG.current_tags,'w')as fw:
			fw.write("\n".join(current_tags))
	
	current_safe_tags = sorted(list(set(current_safe_tags)))
	if current_safe_tags:
		with open(CFG.current_safe_tags,'w')as fw:
			fw.write("\n".join(current_safe_tags))

	current_without_tags = []
	for cwt in safe_tags:
		if cwt not in current_safe_tags:
			current_without_tags.append(cwt)
	current_without_tags = sorted(list(set(current_without_tags)))

	if current_without_tags:
		with open(CFG.current_without_tags,'w')as fw:
			fw.write("\n".join(current_without_tags))

	return all_train_images, all_train_labels




global all_classes
all_classes = load_all_classes()

CFG.num_classes = len(all_classes)
print(f"all_classes: {all_classes[0:10]}")
print(f"num_classes: {len(all_classes)}")

all_train_images, all_train_labels = images2labels(all_classes)
#all_test_images = glob.glob(f'{CFG.test_set_dir}/**/*{CFG.image_extension}')

print("all_train_images:",all_train_images[0:10])
print("all_train_labels:",all_train_labels[0:10])

# for i in range(0,200):
# 	print(f'{all_classes[all_train_labels[i]]} <= {all_train_images[i]}')

#os._exit(0)

dataset_train = mlxDeepDanbooruDataset(all_train_images,all_train_labels, transform, all_classes)
#dataset_test = mlxDeepDanbooruDataset(all_test_images,all_test_labels, transform)


print(f'train images: {len(dataset_train)}')
#print(f'test images: {len(dataset_test)}')

with open("ttttt50.txt","w")as fw:
	fw.write("\n".join(all_train_images[0:50]))

t1 = time.time()


dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = CFG.batch_size, shuffle = True)

print("prepare dataset:",time.time()-t1)