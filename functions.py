import os
import time
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

from typing import List, Tuple, Dict, AnyStr, KeysView, Any

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms

import skimage.transform
import config as CFG 

import matplotlib.pyplot as plt
from tqdm import tqdm




def ToUnixSlash(s):
    return s.replace("\\","/")

def md5str(s):
    fmd5 = hashlib.md5(s.encode('utf-8')).hexdigest().lower()
    return fmd5


def get_model_name(model_instance):
	model_name = type(model_instance).__name__
	print(f'model_name: {model_name}')

	if model_name == "" or model_name is None:
		raise ValueError("ERROR: cannot get the model name.")
	return model_name


def save_append(fpath,data,mode='a'):
	if not os.path.isdir(os.path.dirname(fpath)):
		os.mkdir(os.path.dirname(fpath))
	with open(fpath, mode) as f:
		f.write(data)
		f.close()

def get_latest_file(pattern):
	flist = glob.glob(pattern)
	fmtime = 0
	fname = ""
	for pth in flist:
		pth_mtime = os.path.getmtime(pth)
		if pth_mtime > fmtime:
			fmtime = pth_mtime
			fname = pth
	return fname

def plot_figure(param={'figsize':(30,18),'x':[],'y':[],'title':"",'xlabel':"",'ylabel':"",'savefig_path':""}):
	plt.figure(figsize=param['figsize'])
	plt_x = param['x']
	plt_y = param['y']
	plt.plot(plt_x, plt_y, marker='o')
	plt.title(param['title'])
	plt.xlabel(param['xlabel'])
	plt.ylabel(param['ylabel'])
	plt.xticks(size=22)
	plt.yticks(size=16)
	plt.grid(True)
	lr_dir = os.path.dirname(param['savefig_path'])
	if not os.path.isdir(lr_dir):
		os.makedirs(lr_dir)
	plt.savefig(param['savefig_path'])
	plt.close()

def image_normalize_mean_std(sets=[]):
	if len(sets) == 0:
		return None

	x2 = torch.stack([sample[0] for sample in ConcatDataset(sets)])

	mean = torch.mean(x2, dim=(0,2,3))
	#print("mean:", mean)

	std = torch.std(x2, dim=(0,2,3))
	#print("std:", std)

	return mean, std

def transform_and_pad_image(
    image,
    target_width,
    target_height,
    scale=None,
    rotation=None,
    shift=None,
    order=1,
    mode="edge",
    ):
    """
    Transform image and pad by edge pixles.
    """
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_array = image

    # centerize
    t = skimage.transform.AffineTransform(
        translation=(-image_width * 0.5, -image_height * 0.5)
    )

    if scale:
        t += skimage.transform.AffineTransform(scale=(scale, scale))

    if rotation:
        radian = (rotation / 180.0) * math.pi
        t += skimage.transform.AffineTransform(rotation=radian)

    t += skimage.transform.AffineTransform(
        translation=(target_width * 0.5, target_height * 0.5)
    )

    if shift:
        t += skimage.transform.AffineTransform(
            translation=(target_width * shift[0], target_height * shift[1])
        )

    warp_shape = (target_height, target_width)

    image_array = skimage.transform.warp(
        image_array, (t).inverse, output_shape=warp_shape, order=order, mode=mode
    )

    return image_array






