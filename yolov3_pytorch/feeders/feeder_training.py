import numpy as np
import cv2
import pickle
import sys
import os
import PIL.Image as pilimg
import random
import json

import torch
from torch.utils.data import Dataset

from yolov3_pytorch.utils.detection_util import processing_detection_image_absolute_labels, processing_detection_image_default_augmentation_absolute_labels, processing_detection_image_strong_augmentation_absolute_labels, processing_detection_image_relative_labels, processing_detection_image_default_augmentation_relative_labels, processing_detection_image_strong_augmentation_relative_labels

class FeederTrain(Dataset):
    def __init__(self, data_json_path, object_class_names_list, resize_width_height, image_augmentation, multi_sacle, max_num=-1):
	
        self.data_json_path = data_json_path
        self.object_class_names_list = object_class_names_list
        self.resize_width_height = resize_width_height
        self.image_augmentation = image_augmentation
        self.multi_sacle = multi_sacle
        self.min_size = self.resize_width_height - 3 * 32
        self.max_size = self.resize_width_height + 3 * 32
        self.max_num = max_num
        self.batch_count = 0
        self.load_data()
        
    def load_data(self):
        
        with open(self.data_json_path, 'r') as f:
            	parsing_config = json.load(f)
        
        obj_num = 0
        self.image_path_list = []
        self.label_list = []
        
        self.class_format = parsing_config['class_format']
        self.label_scale = parsing_config['label_scale']
        
        print(f'Class format: {self.class_format}, Label scale: {self.label_scale}')
        
        for i, image in enumerate(parsing_config['image_list']):
            
            # Image path & label
            image_path = image['image_file_path']

            label = parsing_config['object_boxes_list'][i]
            
            object_boxes_num = label['object_box_num']            
            obj_num += object_boxes_num
            		
            self.image_path_list.append(image_path)
            self.label_list.append(label)
            
            if len(self.image_path_list) == self.max_num:
            	print(f'max_num - {i}: break.')
            	break      
        
        print('Aug of Training data: {}'.format(self.image_augmentation)) 
        print('Multi scale training: {}'.format(self.multi_sacle))     
        print('Length of original image list of Training data: {}'.format(parsing_config['image_num']))
        print('Length of original object list of Training data: {}'.format(parsing_config['object_boxes_num']))
        print('Length of filtering image list of Training data: {}'.format(len(self.image_path_list)))
        print('Length of filtering object list of Training data: {}'.format(obj_num))

    def __len__(self):
        return len(self.label_list)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # Image
        try:
            image_path = self.image_path_list[index]
            image = np.array(pilimg.open(image_path))
        except Exception as e:
            print(e)
            print(f"Could not read image. {image_path}")
            return
                    
        # Object boxes
        object_boxes = self.label_list[index]
        object_name_list = object_boxes['object_name_list']
        object_box_list = object_boxes['object_box_list']
        
        # Num of object boxes    
        object_boxes_num = object_boxes['object_box_num']
        
        object_boxes_np = np.zeros((object_boxes_num, 5))
        for a in range(object_boxes_num):
            if self.class_format == 'name':
            	object_boxes_np[a][0] = self.object_class_names_list.index(object_name_list[a])
            elif self.class_format == 'id':
            	object_boxes_np[a][0] = object_name_list[a]
            object_boxes_np[a][1:] = object_box_list[a]
        
        try:
            if self.label_scale == 'absolute':
            	if self.image_augmentation == 'none':
            		image, label = processing_detection_image_absolute_labels(image, object_boxes_np, self.resize_width_height)
            	elif self.image_augmentation == 'default':
            		image, label = processing_detection_image_default_augmentation_absolute_labels(image, object_boxes_np, self.resize_width_height)
            	elif self.image_augmentation == 'strong':
             		image, label = processing_detection_image_strong_augmentation_absolute_labels(image, object_boxes_np, self.resize_width_height)         	
            elif self.label_scale == 'relative':
            	if self.image_augmentation == 'none':
            		image, label = processing_detection_image_relative_labels(image, object_boxes_np, self.resize_width_height)
            	elif self.image_augmentation == 'default':
            		image, label = processing_detection_image_default_augmentation_relative_labels(image, object_boxes_np, self.resize_width_height)
            	elif self.image_augmentation == 'strong':
             		image, label = processing_detection_image_strong_augmentation_relative_labels(image, object_boxes_np, self.resize_width_height)
             		
        except Exception as e:
            print(e)
            print(f"Could not apply transform. {image_path}")
            return
        
        return image, label, object_boxes_num, image_path
        
    def collate_fn(self, batch):
        
        self.batch_count += 1
        
        # Drop invalid images
        batch = [data for data in batch if data is not None]
        
        imgs, labels, object_boxes_num, image_path = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multi_sacle and self.batch_count % 10 == 0:
            self.resize_width_height = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        
        # Stack images
        imgs = torch.stack([img for img in imgs])

        # Add sample index to targets
        for i, label in enumerate(labels):
            label[:, 0] = i
        labels = torch.cat(labels, 0)
        
        return imgs, labels, image_path
