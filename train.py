import random
import os
import numpy as np
import time
from tqdm import tqdm
import ast
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from yolox.exp import get_exp
from detection_util import load_classes

class Yolox_Train_Processor():
	def __init__(self, config_dict, device_name, random_seed_num):
		
		print('-'*60)
		
		## Configs
		self.config_dict = config_dict		
		yolox_config = self.config_dict['yolox_config']
		train_config = self.config_dict['train_config']
		
		# Device
		if torch.cuda.is_available() and device_name.find('cuda') != -1:
			self.device = torch.device(device_name)
			torch.backends.cudnn.deterministic = True
			torch.cuda.set_device(self.device)
		else:
			self.device = torch.device('cpu')
		print('Device:', self.device)
		
		# Params
		model_name = yolox_config.get('yolox', 'model_name')
		self.dataset_name = train_config.get('data', 'dataset_name')
		data_dir = train_config.get('data', 'data_dir')
		self.resize_width_height = yolox_config.getint('yolox', 'resize_width_height')
		img_size = (self.resize_width_height, self.resize_width_height)
		print('Model name:', model_name)
		print('Data folder:', data_dir)
		print('Resize width, height:', self.resize_width_height)

		# Object class names
		object_class_names_list_path = yolox_config.get('class_names', 'object_class_names_list_path', raw=True)
		self.object_class_names_list = load_classes(object_class_names_list_path) # e.g. ['person', ...]
		self.class_num = len(self.object_class_names_list)
		print('Number of names:', self.class_num)
		print('Object class names list:', self.object_class_names_list)

		# Yolox func
		self.exp = get_exp(self.dataset_name, data_dir, self.object_class_names_list, img_size, exp_name=model_name)
		
		print('-'*60)	
		
	def train_eval(self):
		
		trainer = self.exp.get_trainer(self.config_dict, self.device, self.dataset_name, self.object_class_names_list, self.class_num, self.resize_width_height)
		trainer.train()
			
