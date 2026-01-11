#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

import config as CFG
from functions import *
from utils import *
from dataset import *
from TorchDeepDanbooru.deep_danbooru_model import *





#model = mlxDeepDanbooruNet()
model = DeepDanbooruModel()
model.tags = load_all_classes()

model_name = get_model_name(model)
model_output_dir = f'output/{model_name}'
print(model_output_dir)
if not os.path.isdir(model_output_dir):
	os.makedirs(model_output_dir)

num_epoch = 2000
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)




accumulator = train_epoch(model, loss_fn, dataloader_train, optimizer, num_epoch, model_output_dir)
print("------------------")
#print(accumulator)

