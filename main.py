import argparse
import numpy as np
import PIL.Image as pilimg
import cv2
from configparser import ConfigParser

from detect_image import Yolox_Image_Dection_Processor
from detection_util import draw_boxes
from train import Yolox_Train_Processor

def parse_args():
	parser = argparse.ArgumentParser(
		description='Yolo-x model train/inference')
	
	# Run mode
	parser.add_argument('--mode', help='yolox run mode', choices=['yolox-detection-img', 'yolox-train'], default='yolox-detection-img')
	
	# CUDA or not
	parser.add_argument('--device', help='device of yolox (cpu, cuda:0, ...)', default='cuda:0')

	# Random seed
	parser.add_argument('--random_seed_num', help='number of random seed', type=int, default=99)
	
	# Config file path
	parser.add_argument('--yolox_config_file_path', help='config file of yolox', default='./config/yolox_config.ini')
	parser.add_argument('--train_config_file_path', help='config file of train', default='./config/train_config.ini')
	parser.add_argument('--tensorrt_config_file_path', help='config file of tensorrt', default='./config/tensorrt_config.ini')
	
	# Inference image & save path
	parser.add_argument('--inf_img_path', help='inference image path', default='./datasets/test_image/1066405,1b8000ef60354f.jpg')
	parser.add_argument('--save_path', help='save path of inference image', default='./datasets/test_image')
		
	args = parser.parse_args()
	return args

def read_config(path):
	config = ConfigParser()
	config.read(path, encoding='utf-8') 
	
	return config

def get_config_dict(config_path_list):
	
	config_dict = {}
	
	for config_path in config_path_list:
		config_dict[config_path.split('/')[-1].split('.')[0]] = read_config(config_path)
	
	return config_dict
	
def main():
	# Arg parsing
	args = parse_args()
	
	# Device
	device = args.device
	
	# Random seed
	random_seed_num = args.random_seed_num
	
	# Config dict
	config_dict = get_config_dict([args.yolox_config_file_path, args.train_config_file_path, args.tensorrt_config_file_path])
	
	if args.mode == 'yolox-detection-img':
		
		# Read image
		print('Inf image path:', args.inf_img_path)
		img = cv2.imread(args.inf_img_path, cv2.IMREAD_COLOR)
		print('Size of read image:', img.shape) # (h, w, c)
	
		# Yolox inference
		yolox_image_detection_processor = Yolox_Image_Dection_Processor(config_dict, device)
		box_list, class_list, conf_list, num_objects = yolox_image_detection_processor.predict(img)
		
		print('-'*30)
		print('Object box list:', box_list)
		print('Class name list:', class_list)
		print('Conf score list:', conf_list)
		print('-'*30)
		
		# Draw objects
		draw_boxes(img, box_list, class_list, args.inf_img_path, args.save_path)

	elif args.mode == 'yolox-train':
		yolox_train_processor = Yolox_Train_Processor(config_dict, device, random_seed_num)	
		yolox_train_processor.train_eval()
	
if __name__ == '__main__':
	main()
