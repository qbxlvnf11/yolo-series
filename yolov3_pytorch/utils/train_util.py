import ast
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable

class TrainSettings():
	
	def __init__(self, train_config):
		
		# Train
		self.epochs = train_config.getint('train', 'epochs')	
		self.num_workers = train_config.getint('train', 'num_workers')	
		self.fine_tune_mode = train_config.getboolean('train', 'fine_tune_mode')
		self.image_augmentation = train_config.get('train', 'image_augmentation')
		self.multi_sacle_train_mode = train_config.getboolean('train', 'multi_sacle_train_mode')
		
		# Data
		self.batch_size = train_config.getint('data', 'batch_size')
		self.train_data_json_path = train_config.get('data', 'train_data_json_path', raw=True)
		self.valid_data_json_path = train_config.get('data', 'valid_data_json_path', raw=True)
		
		# Save model
		self.save_weight_mode = train_config.getboolean('save', 'save_weight_mode')
		self.save_eval_interval = train_config.getint('save', 'save_eval_interval')
		folder_path = train_config.get('save', 'folder_path', raw=True)
		weight_name = train_config.get('save', 'weight_name')
		self.save_path = os.path.join(folder_path, weight_name)
		
		# Optimizer
		self.optimizer_name = train_config.get('optimize', 'optimizer')
		self.momentum = train_config.getfloat('optimize', 'momentum')
		self.nesterov_mode = train_config.getboolean('optimize', 'nesterov_mode')
		self.base_lr = train_config.getfloat('optimize', 'base_lr')
		self.lr_decay = train_config.getfloat('optimize', 'lr_decay')
		self.weight_decay = train_config.getfloat('optimize', 'weight_decay')
		self.lr_decay_step = ast.literal_eval(train_config.get('optimize', 'lr_decay_step'))
	
	def select_loss_function(self, device):
		self.loss = nn.CrossEntropyLoss().to(device)
		print('Loss function:', self.loss)
	
	def select_optimizer(self, model):
		if self.optimizer_name == 'SGD':
			self.optimizer = optim.SGD(
			model.parameters(),
			lr=self.base_lr,
			momentum=self.momentum,
			nesterov=self.nesterov_mode,
			weight_decay=self.weight_decay)
		elif self.optimizer_name == 'Adam':
			self.optimizer = optim.Adam(
			model.parameters(),
			lr=self.base_lr,
			weight_decay=self.weight_decay)
		else:
			raise ValueError()
		print('Optimizer:', self.optimizer)
		
		''''
		scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
                '''
		
		self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_step, gamma=self.lr_decay)
		
