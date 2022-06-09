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

from yolov3_pytorch.feeders.feeder_training import FeederTrain
from yolov3_pytorch.feeders.feeder_validating import FeederValid
from yolov3_pytorch.utils.train_util import TrainSettings
from yolov3_pytorch.utils.detection_util import load_classes, non_max_suppression
from yolov3_pytorch.yolov3_model import Darknet as Yolov3
from yolov3_pytorch.utils.yolov3_def_parser import parse_yolov3_def
from yolov3_pytorch.utils.yolo_v3_util import compute_loss, get_batch_statistics, ap_per_class, print_eval_stats

class Yolov3_Train_Processor():
	def __init__(self, config_dict, device_name, random_seed_num):
		
		print('-'*60)
		
		## Configs
		yolov3_config = config_dict['yolov3_config']
		train_config = config_dict['train_config']

		# Train settings
		self.train_settings = TrainSettings(train_config)

		# Init seed
		self.random_seed_num = random_seed_num
		self.__init_seed(self.random_seed_num)

		# Device
		if torch.cuda.is_available() and device_name.find('cuda') != -1:
			self.device = torch.device(device_name)
			torch.backends.cudnn.deterministic = True
			torch.cuda.set_device(self.device)
		else:
			self.device = torch.device('cpu')
		print('Device:', self.device)
		
		# Yolov3 param
		yolov3_channels = yolov3_config.getint('yolov3', 'yolov3_channels')
		model_def_config_path = yolov3_config.get('yolov3', 'model_def_config_path', raw=True)
		self.module_defs = parse_yolov3_def(model_def_config_path)
		#print('Module defs:', self.module_defs)

		# Yolov3 weights path
		self.weight_path = yolov3_config.get('yolov3', 'weight_path', raw=True)
		
		# Load object class name list
		self.object_class_names_list = load_classes(yolov3_config.get('class_names', 'object_class_names_list_path', raw=True))

		# Input img size
		self.resize_width_height = yolov3_config.getint('yolov3', 'resize_width_height')	

		# Threshold
		self.object_confidence_threshold = yolov3_config.getfloat('threshold', 'object_confidence_threshold')
		self.nms_thres = yolov3_config.getfloat('threshold', 'nms_thres')
		self.IoU_threshold = yolov3_config.getfloat('threshold', 'IoU_threshold')
			
		# Init
		self.start_epoch = 0
		self.best_mAP = 0
		
		# Load data
		self.__load_data()
		
		# Init yolov3
		self.model = Yolov3(self.module_defs, yolov3_channels).to(self.device)
		self.model.apply(self.__weights_init_normal)
		
		# Load pretrained weight
		self.__load_model()
		
		# Selecting loss function & Optimizer
		self.train_settings.select_loss_function(self.device)
		self.train_settings.select_optimizer(self.model)
	
	def __init_seed(self, random_seed_num):
		#print('Seed setting ... random seed num:', random_seed_num)
		torch.cuda.manual_seed_all(random_seed_num)
		torch.manual_seed(random_seed_num)
		np.random.seed(random_seed_num)
		random.seed(random_seed_num)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False		
	
	def __load_data(self):
		print('Train data json path:', self.train_settings.train_data_json_path)
		print('Valid data json path:', self.train_settings.valid_data_json_path)
		
		train_feeder = FeederTrain(data_json_path=self.train_settings.train_data_json_path, object_class_names_list=self.object_class_names_list, resize_width_height=self.resize_width_height, image_augmentation=self.train_settings.image_augmentation, multi_sacle=self.train_settings.multi_sacle_train_mode)
		valid_feeder = FeederValid(data_json_path=self.train_settings.valid_data_json_path, object_class_names_list=self.object_class_names_list,  resize_width_height=self.resize_width_height)
		
		self.data_loader = dict()
		
		self.data_loader['train'] = torch.utils.data.DataLoader(
			dataset=train_feeder,
			collate_fn=train_feeder.collate_fn,
			batch_size=self.train_settings.batch_size,
			shuffle=True,
			num_workers=self.train_settings.num_workers,
			drop_last=False,
			worker_init_fn=self.__init_seed(self.random_seed_num))
		self.data_loader['valid'] = torch.utils.data.DataLoader(
			dataset=valid_feeder,
			collate_fn=valid_feeder.collate_fn,
			batch_size=self.train_settings.batch_size,
			shuffle=False,
			num_workers=self.train_settings.num_workers,
			drop_last=False,
			worker_init_fn=self.__init_seed(self.random_seed_num))
		
		print('-'*60)

	def __weights_init_normal(self, m):
		classname = m.__class__.__name__
		if classname.find("Conv") != -1:
			nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find("BatchNorm2d") != -1:
			nn.init.normal_(m.weight.data, 1.0, 0.02)
			nn.init.constant_(m.bias.data, 0.0)
        
	def __load_model(self):
		
		if self.train_settings.fine_tune_mode:

			print('Pretrained Yolov3 model load ...')
			print('Load weights from {}.'.format(self.weight_path))
			
			# Load pretrained model    
			if self.weight_path.endswith(".pt"):
				# Checkpoint
				checkpoint = torch.load(self.weight_path, map_location=self.device)
				self.model.load_state_dict(checkpoint)			
			else:
				# Weights
				self.model.load_darknet_weights(self.weight_path)

			
	def __train(self, epoch, save_model):
		self.model.train()
		print('-'*30)
		print('Training epoch: {}'.format(epoch + 1))

		# Learning rate
		print('Learning rate:', self.train_settings.optimizer.param_groups[0]['lr'])
		
		# Train loss list
		loss_list = []
		
		# Load train data
		loader = self.data_loader['train']
		process = tqdm(loader)
		
		for batch_id, (data, label, image_path) in enumerate(process):
			
			# Image
			data = data.to(self.device) # e.g. (16, 3, 416, 416)
			
			# Lable: number of labels / image_id, class, relative_scale(center_x, center_y, w, h)
			label = label.to(self.device) # e.g. (100, 6)
			
			# Reset gradients
			self.train_settings.optimizer.zero_grad()	
					
			# Yolov3 forward outputs: 1 / number of proposals / (center_x, center_y, w, h, conf) + number of class names
			output = self.model(data) # e.g. (1, 10647, 85)
			
			# Compute loss
			loss, loss_components = compute_loss(output, label, self.model)
			
			# Print loss
			if batch_id % 100 == 0:
				print('='*50)
				print(f"IoU loss: {float(loss_components[0]):.6f}, Object loss: , {float(loss_components[1]):.6f}, Class loss: {float(loss_components[2]):.6f} \nTotal Loss: {float(loss_components[3])}")
				print('='*50)	
			
			# Yolov3 backward
			loss.backward()
			loss_list.append(loss.data.item())
			
			# Optimizing
			self.train_settings.optimizer.step()

		print('Mean training loss: {:.6f}.'.format(np.mean(loss_list)))

		# Learning rate scheduling
		self.train_settings.lr_scheduler.step()
		
		if save_model:
			
			if not os.path.exists(self.train_settings.save_path):
				print('Make folder {}'.format(self.train_settings.save_path))
				os.makedirs(self.train_settings.save_path)
			torch.save(self.model.state_dict(), os.path.join(self.train_settings.save_path, str(epoch+1) + '.pt'))
			print('Complete to save model')	

	def __eval(self, epoch):
		self.model.eval()
		print('-'*30)
		print('Validation epoch: {}'.format(epoch + 1))
		
		# Load train data
		loader = self.data_loader['valid']
		process = tqdm(loader)
		
		class_list = []
		sample_metrics = []
		total_detected_boxes_len = 0
		
		for batch_idx, (data, label, image_path) in enumerate(process):	

			with torch.no_grad():
				
				# Image
				data = data.to(self.device) # e.g. (16, 3, 416, 416)
				
				# Lable: number of labels / image_id, class, relative_scale(center_x, center_y, w, h)
				label = label.to(self.device) # e.g. (100, 6)
				label[:, 2:] *= self.resize_width_height 
				
				# Yolov3 outputs: 1 / number of proposals / (center_x, center_y, w, h, conf) + number of class names
				output = self.model(data) # e.g. (16, 10647, 85)
				
				# NMS (Non maximum suppression) outputs: 1 / number of objects to detect / absolute_scale(x, y, x, y), score, class
				detections = non_max_suppression(output, conf_thres=self.object_confidence_threshold, iou_thres=self.nms_thres, xywh=True) # e.g. (1, 3, 6)
				
				# Class list
				class_list += copy.deepcopy(label).cpu().numpy()[:, 1].tolist()
			
			# Batch statistics
			batch_metrics, detected_boxes_len = get_batch_statistics(detections, label, iou_threshold=self.IoU_threshold)
			sample_metrics += batch_metrics
			total_detected_boxes_len += detected_boxes_len
							
		if len(sample_metrics) == 0:  # No detections over whole validation set.
			print("---- No detections over whole validation set ----")
			return None
		
		class_list = np.array(class_list)
		class_unique_list = np.unique(class_list)
		
		# Concatenate sample statistics
		true_positives, pred_scores, pred_classes = [
			np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
			
		metrics_output = ap_per_class(
			true_positives, pred_scores, pred_classes, class_list)
		
		# mAP
		mAP = print_eval_stats(metrics_output, self.object_class_names_list)

		if mAP > self.best_mAP:
			self.best_mAP = mAP
			self.best_epoch = epoch + 1
		
	def train_eval(self):
			
		for epoch in range(self.start_epoch, self.train_settings.epochs):
			save_eval_model_flag = (self.train_settings.save_weight_mode and ((epoch + 1) % self.train_settings.save_eval_interval == 0)) or (self.train_settings.save_weight_mode and ((epoch + 1) == self.train_settings.epochs))

			self.__train(epoch, save_model=save_eval_model_flag)
			
			if save_eval_model_flag:
				self.__eval(epoch)

		print('Best mAP: {}, Epochs: {}'.format(self.best_mAP, self.best_epoch))
			
