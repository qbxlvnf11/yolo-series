### Format of parsing data dict
# parsing_data_dic['class_format'] = type of class ('name' or 'id')
# parsing_data_dic['label_scale'] = scale of label ('absolute' or 'relative')
# parsing_data_dic['image_list'] = [{'id'-image id, 'image_file_path'-image file path}, ...]
# parsing_data_dic['object_boxes_list'] = [{'image_id'-image id, 'object_box_num'-number of the object per image, 'object_box_id_list'-[object box id, ...], 'object_name_list'-[object name, ...], 'object_box_list'-[[center x, center y, box_width, box_height], ...], 'object_box_size_list'-[object box size, ...], }, ...]
# parsing_data_dic['image_num'] = number of the image
# parsing_data_dic['object_boxes_num'] = [number of the total objects, ...]

import argparse
import json
import os
import pickle
import PIL.Image as pilimg
import numpy as np

from data.data_parser.detection_coco2014_dataset_parser import detection_coco2014_dataset_parsing

def build_detection_data_json_file(parsing_data_dic, save_folder_path, save_file_name):
	
	print('Save folder path:', save_folder_path)	
	
	for key in parsing_data_dic.keys():
		
		json_save_file_name = save_file_name + '_' + key + '.json'
		print('Save file name:', json_save_file_name)

		if not os.path.exists(save_folder_path):
			print('Make folder {}'.format(save_folder_path))
			os.makedirs(save_folder_path)
		
		with open(os.path.join(save_folder_path, json_save_file_name), 'w') as f:
			json.dump(parsing_data_dic[key], f)
			print('Save complete')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--target', help='target data', choices=['coco2014'], default='coco2014')
	parser.add_argument('--load_data_mode', help='loading image folder or data path list', choices=['image_folder', 'data_path_txt'], default='data_path_txt')
	parser.add_argument('--data_folder_path', help='target data folder path', default='./data/train_data/coco')
	parser.add_argument('--save_folder_path', help='save folder path', default='./data/data_json/coco2014')
	parser.add_argument('--save_file_name', help='parsing data dict file save path', default='coco2014')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	
	parsing_data_dic = {}
	
	if args.target == 'coco2014':
		parsing_data_dic['train'] = detection_coco2014_dataset_parsing(mode='train2014', data_folder_path=args.data_folder_path, load_data_mode=args.load_data_mode, data_path_txt_name='trainvalno5k.txt')
		parsing_data_dic['valid'] = detection_coco2014_dataset_parsing(mode='val2014', data_folder_path=args.data_folder_path, load_data_mode=args.load_data_mode, data_path_txt_name='5k.txt')
		
	build_detection_data_json_file(parsing_data_dic, args.save_folder_path, args.save_file_name)

if __name__ == '__main__':
    main()
