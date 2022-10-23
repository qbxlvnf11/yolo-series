import time
import tqdm
import numpy as np
import cv2
import collections
import os

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torchvision
from torchvision.ops import nms as nms
import torchvision.transforms as transforms
import torch.nn.functional as F

def load_classes(path):
	"""
	Loads class labels at 'path'
	"""
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

def filter_classes(detections, classes):
	mask = torch.stack([torch.stack([detections[:, -1] == cls]) for cls in classes])
	mask = torch.sum(torch.squeeze(mask, dim=1), dim=0)
	return detections[mask > 0]

def scale_coords(detections, ratio):
	
	detections = detections.cpu()
	bboxes = detections[:, 0:4]

	# Rescale coords (xyxy)
	bboxes /= ratio
	
	return bboxes
	
def draw_boxes(img, box_list, class_list, inf_img_path, save_path):
	
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	color = (0, 0, 255) # red
	thickness = 2 # thickness of line
	
	for box, class_name in zip(box_list, class_list):
		# xyxy format
		x1, y1, x2, y2 = list(map(int, box))

		t_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
		cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
		cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
		cv2.putText(img, class_name, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
	
	name = inf_img_path.split('/')[-1].split('.')[0]+'_inferencing_yolox.png'

	if not os.path.exists(save_path):
		print('Make folder {}'.format(save_path))
		os.makedirs(save_path)
	cv2.imwrite(os.path.join(save_path, name), img)
	
	print(img.shape)
	print('Complete save inferencing image!', os.path.join(save_path, name))

