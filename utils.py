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

import config as CFG 
from functions import *
from dataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm


def train_epoch(net, loss_fn, train_iter, optimizer, epochs,output_dir):
	device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	print('train on device:', device)
	net = net.to(device)
	epoch_start = 0
	model_path = None
	latest_pth = get_latest_file(f'{output_dir}/*.pth')
	if latest_pth !="":
		epoch_start = int(os.path.basename(latest_pth).replace("model_","").replace(".pth","").strip())
		model_path = latest_pth
	print(f'epoch_start: {epoch_start}')

	if model_path is not None:
		ckpt = torch.load(model_path, map_location=device)
		net.load_state_dict(ckpt)
		epochs = epochs + epoch_start
	#return None

	one_hot_f = nn.functional.one_hot
	epoch_loss = []
	train_loss = []
	last_loss = 0.0
	time_train_start = 0
	time_vali_start = 0
	batch_count = len(train_iter)
	print('batch_count:',len(train_iter))
	for epoch in range(epoch_start,epochs):
		time_train_start = time.time()
		len_train = 0
		len_vali = 0

		net.train()
		epoch_loss.clear()	
		
		for img, label in train_iter:
			#print(f'img:{img[0:10]},{img.shape}')
			#print(f'label:{label[0:10]},{label.shape}')
			img = img.to(device,dtype=torch.float32)
			label = label.to(device)
			optimizer.zero_grad()
			y_hat = net(img)
			#print(f'y_hat:{y_hat},{y_hat.shape}')
			loss = loss_fn(y_hat, label)
			#print(f'loss: {loss}')
			loss.backward()
			optimizer.step()
			#
			epoch_loss.append(loss.item())

			if len(epoch_loss) > 0:
				max_print = 2
				if len(epoch_loss) < max_print:
					max_print = len(epoch_loss)
				print(f'{len(epoch_loss)}/{epoch} loss: {loss} | epoch_loss: {sum(epoch_loss)/len(epoch_loss)} | {epoch_loss[0:max_print]}...\r', end="")

		train_loss.append(sum(epoch_loss)/len(epoch_loss))
		if len(train_loss) > 0:
			max_print = 3
			if len(train_loss) < max_print:
				max_print = len(train_loss)
			print(f'train_loss: {sum(train_loss)/len(train_loss)} | {train_loss[0:max_print]}...\r')

		plot_figure(param = {
			'figsize': (10,6),
			'x': range(epoch_start, epoch_start+len(train_loss)),
			'y': train_loss,
			'title': f'Loss Function Curve - Epoch: {epoch+1}',
			'xlabel': f'Batch: 0 - { len(train_loss) }, Batch Size: {CFG.batch_size}',
			'ylabel': 'Loss',
			'savefig_path': f'{output_dir}/lr_{ optimizer.state_dict()["param_groups"][0]["lr"] }/1_loss_of_epoch_{epoch_start}-{epoch+1}.png'
			})
	

		print(f'----------- epoch: {epoch+1} start --------------')
		print(f'epoch: {epoch+1} / {epochs} train loss: { sum(train_loss)/len(train_loss) }')
		print(f'epoch: {epoch+1} / {epochs} train time: {int(time.time()-time_train_start)} sec')

			# save
		if (((epoch+1) % 3 == 0) or (epoch + 1 >= epochs)):
			state_dict = net.state_dict()
			state_dict["tags"] = load_all_classes()
			torch.save(state_dict, f'{output_dir}/model_{str(epoch+1)}.pth')

	#return accumulator








