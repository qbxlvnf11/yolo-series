import argparse
import cv2
import json
import os
import numpy as np
import PIL.Image as pilimg
from xml.etree import ElementTree

def detection_coco2014_dataset_parsing(mode, data_folder_path, load_data_mode, data_path_txt_name):
    print('-'*50)
    print('Mode:', mode)
    print(f'COCO data folder path: {data_folder_path}')
    
    # Parsing data dict
    coco214_data_dict = {}
    
    label_dir_path = os.path.join(data_folder_path, 'labels', mode)
    img_dir_path = os.path.join(data_folder_path, 'images', mode)
    print('label_dir_path:', label_dir_path)
    print('img_dir_path:', img_dir_path)
    
    if load_data_mode == 'image_folder':    
            # Image names
            img_names = os.listdir(img_dir_path)
    elif load_data_mode == 'data_path_txt':    
            # Image names
            img_names = []
            # Selecting train or val data
            mode_list = []
            with open(os.path.join(data_folder_path, data_path_txt_name), mode='r', encoding='utf-8') as file:
            	while True:
            		line = file.readline()
            		if not line: break
            		line_split = line.rstrip().rstrip('\n').split('/')
            		img_names.append(line_split[-1])
            		mode_list.append(line_split[-2])
    
    print('length of images:', len(img_names))
    
    # Init
    object_box_id = 0
    image_list = []
    object_boxes_list = []
        
    for image_id, img_name in enumerate(img_names):
        	
            if image_id % 1000 == 0:
            	print('image_id:', image_id)
            
            ## Image dict
            image = {}

            # Image id
            image['id'] = image_id

            # Image path
            if load_data_mode == 'image_folder': 
            	image['image_file_path'] = os.path.join(img_dir_path, img_name)
            elif load_data_mode == 'data_path_txt': 
            	image['image_file_path'] = os.path.join(data_folder_path, 'images', mode_list[image_id], img_name)   
            
            ## Object box dict
            object_box = {}

            # Image id
            object_box['image_id'] = image_id
             
            # Read line of label file
            if load_data_mode == 'image_folder': 
            	label_path = os.path.join(label_dir_path, img_name.split('.')[0] + '.txt')
            elif load_data_mode == 'data_path_txt':             
            	label_path = os.path.join(data_folder_path, 'labels', mode_list[image_id], img_name.split('.')[0] + '.txt')

            class_list = []
            box_list = []
            
            # class
            try:	
            	with open(label_path, mode='r', encoding='utf-8') as file:
            		while True:
            			line = file.readline()
            			if not line: break
            			line_split = line.rstrip().rstrip('\n').split(' ')
            			class_list.append(int(line_split[0]))
            			box_list.append(list(map(float, line_split[1:])))
            			
            except Exception as e:
            	print(e)
            	continue
            
            # Number of the objects per images
            object_box['object_box_num'] = len(class_list)
            
            # Object box ids
            object_box['object_box_id_list'] = []
            
            # Object names
            object_box['object_name_list'] = []
            
            # Object boxes
            object_box['object_box_list'] = []

            # Object box sizes
            object_box['object_box_size_list'] = []
			
            for obj_id, (class_num, box) in enumerate(zip(class_list, box_list)):
            	
            	# Object box id
            	object_box['object_box_id_list'].append(object_box_id)
	            
            	# Object id
            	object_box['object_name_list'].append(class_num)	
            	
            	# center_x/center_y/box_w/box_h format format
            	center_x = box[0]
            	center_y = box[1]
            	box_width = box[2]
            	box_height = box[3]
            	box = [center_x, center_y, box_width, box_height]
            	box_size = box_width * box_height 
	            
            	object_box['object_box_list'].append(box)
            	object_box['object_box_size_list'].append(box_size)
	            
            	object_box_id += 1
            
            image_list.append(image)	
            object_boxes_list.append(object_box)
            
    coco214_data_dict['class_format'] = 'id'
    coco214_data_dict['label_scale'] = 'relative'           
    coco214_data_dict['image_list'] = image_list
    coco214_data_dict['image_num'] = len(image_list)
    coco214_data_dict['object_boxes_list'] = object_boxes_list
    coco214_data_dict['object_boxes_num'] = object_box_id

    print('Num of images:', coco214_data_dict['image_num'])
    print('Num of boxes:', coco214_data_dict['object_boxes_num'])

    return coco214_data_dict

