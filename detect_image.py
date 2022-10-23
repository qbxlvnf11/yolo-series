import torch
import numpy as np
import cv2
import os
import PIL.Image as pilimg
import matplotlib
import ast
import time
from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
from detection_util import load_classes, filter_classes, scale_coords

import tensorrt as trt
from cuda import cuda 
import pycuda.autoinit

TRT_LOGGER = trt.Logger()

class Yolox_Image_Dection_Processor():
	def __init__(self, config_dict, device_name):
		
		print('-'*60)
		
		## Configs
		yolox_config = config_dict['yolox_config']
		tensorrt_config = config_dict['tensorrt_config']
		
		# Device
		if torch.cuda.is_available() and device_name.find('cuda') != -1:
			self.device = torch.device(device_name)
			torch.backends.cudnn.deterministic = True
			torch.cuda.set_device(self.device)
		else:
			self.device = torch.device('cpu')
		print('Device:', self.device)
		
		# Yolox model config
		model_name = yolox_config.get('yolox', 'model_name')
		self.weight_path = yolox_config.get('yolox', 'weight_path', raw=True)
		print('Model name:', model_name)

		# Threshold
		self.object_confidence_threshold = yolox_config.getfloat('threshold', 'object_confidence_threshold')
		self.nms_thres = yolox_config.getfloat('threshold', 'nms_thres')
		
		# Input img size
		self.resize_width_height = yolox_config.getint('yolox', 'resize_width_height')

		# Filtering object class names
		self.filtering_img_mode = yolox_config.getboolean('class_names', 'filtering_img_mode')
		target_detection_class_names = yolox_config.get('class_names', 'target_detection_class_names')
		target_detection_class_names = ast.literal_eval(target_detection_class_names) # e.g. ['person']
		object_class_names_list_path = yolox_config.get('class_names', 'object_class_names_list_path', raw=True)
		self.object_class_names_list = load_classes(object_class_names_list_path) # e.g. ['person', ...]
		self.class_num = len(self.object_class_names_list)
		#print('Object class names list:', self.object_class_names_list)
		
		self.classes_index_list = []
		for i, c in enumerate(self.object_class_names_list):
			if c in target_detection_class_names:
				self.classes_index_list.append(i)

		# TensorRT
		self.tensorrt_mode = tensorrt_config.getboolean('tensorrt', 'tensorrt_mode')
		if self.tensorrt_mode:
			self.tensorrt_engine_path = tensorrt_config.get('tensorrt', 'tensorrt_engine_path', raw=True)
			
			# Read the engine from the file and deserialize
			with open(self.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
				self.engine = runtime.deserialize_cuda_engine(f.read())    

			self.context = self.engine.create_execution_context()
			self.context.set_binding_shape(0, (1, 3, self.resize_width_height, self.resize_width_height))

		# Yolox func
		exp = get_exp(None, None, self.object_class_names_list, self.resize_width_height, None, model_name)
		self.preproc = ValTransform(legacy=False)
			
		# Get Yolox
		self.model = exp.get_model().to(self.device)
		logger.info("Model Summary: {}".format(get_model_info(self.model, (self.resize_width_height, self.resize_width_height))))
		
		# Load pretrained weight
		self.__load_model()
		
		print('-'*60)
	
	def __load_model(self):
		
		print('Detection model load ...')
		print('Load weights from {}.'.format(self.weight_path))

		checkpoint = torch.load(self.weight_path, map_location=self.device)

		# Load pretrained model
		self.model.load_state_dict(checkpoint["model"])

		self.model.eval()

	def __trt_inference(self, data):  
	    
		nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
		nOutput = self.engine.num_bindings - nInput

		for i in range(nInput):
			print("Bind[%2d]:i[%2d]->" % (i, i), self.engine.get_binding_dtype(i), self.engine.get_binding_shape(i), self.context.get_binding_shape(i), self.engine.get_binding_name(i))
		for i in range(nInput,nInput+nOutput):
			print("Bind[%2d]:o[%2d]->" % (i, i - nInput), self.engine.get_binding_dtype(i), self.engine.get_binding_shape(i), self.context.get_binding_shape(i), self.engine.get_binding_name(i))

		bufferH = []
		bufferH.append(np.ascontiguousarray(data.reshape(-1)))

		for i in range(nInput, nInput + nOutput):
			bufferH.append(np.empty(self.context.get_binding_shape(i), dtype=trt.nptype(self.engine.get_binding_dtype(i))))

		bufferD = []
		for i in range(nInput + nOutput):
			bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

		for i in range(nInput):
			cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)

		self.context.execute_v2(bufferD)

		for i in range(nInput, nInput + nOutput):
			cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)

		for b in bufferD:
			cuda.cuMemFree(b)  

		return bufferH
	
	def predict(self, inp_img): # e.g. (640, 960, 3)
		
		# Preprocessing input image: e.g. (3, 640, 640)
		img, _ = self.preproc(inp_img, None, (self.resize_width_height, self.resize_width_height))
		
		# Expand dims: e.g. (1, 3, 640, 640)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.float().to(self.device)
		ratio = min(img.shape[2] / inp_img.shape[0], img.shape[3] / inp_img.shape[1])
		
		print('Size of processing image:', img.shape)
		print('Ratio:', ratio)
		
		with torch.no_grad():
			
			start_time = time.time()
			
			if not self.tensorrt_mode:
				
				print('Pytorch Model Inference')
				
				# Yolox outputs: 1 / number of proposals / (boxes, conf) + number of class names. e.g. (1, 8400, 85)
				outputs = self.model(img)
				
			else:
				
				print('TensorRT Engine Inference')
				
				# TensorRT inference
				trt_outputs = self.__trt_inference(img.cpu().numpy())
				
				# Yolox outputs: 1 / number of proposals / (boxes, conf) + number of class names. e.g. (1, 8400, 85)
				outputs = np.array(trt_outputs[1])
				outputs = torch.tensor(outputs).to(self.device)
				self.model.head.decode_outputs(outputs, dtype = outputs[0].type(), hw = [torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])])
				
			# Postprocessing: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
			detections = postprocess(outputs, self.class_num, self.object_confidence_threshold, self.nms_thres)[0]
			
			# Filtering classes
			if self.filtering_img_mode:
				detections = filter_classes(detections, self.classes_index_list)
							
			# Get object boxes
			box_list = [bb for bb in scale_coords(detections, ratio)]
			
			# Get object class names
			class_list = [self.object_class_names_list[int(d[6])] for d in detections]
			
			# Get conf scores
			conf_list = [d[4] for d in detections]

			# Number of objects
			num_objects = len(box_list)	
			
			end_time = time.time()

			print('Yolox inf time:', end_time - start_time)
					
			return box_list, class_list, conf_list, num_objects
			
