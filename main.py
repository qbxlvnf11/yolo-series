import argparse
import numpy as np
import PIL.Image as pilimg
from configparser import ConfigParser

from detect_image import Yolov3_Image_Dection_Processor
from train import Yolov3_Train_Processor
from yolov3_pytorch.utils.detection_util import draw_boxes

def parse_args():
	parser = argparse.ArgumentParser(
		description='Yolo-v3 model train/inference')
	
	# Run mode
	parser.add_argument('--mode', help='yolov3 run mode', choices=['yolov3-detection-img', 'yolov3-train'], default='yolov3-detection-img')
	
	# CUDA or not
	parser.add_argument('--device', help='device of yolov3 (cpu, cuda:0, ...)', default='cuda:0')

	# Random seed
	parser.add_argument('--random_seed_num', help='number of random seed', type=int, default=99)
	
	# Config file path
	parser.add_argument('--yolov3_config_file_path', help='config file of yolov3', default='./config/yolov3_config.ini')
	parser.add_argument('--train_config_file_path', help='config file of train param', default='./config/train_config.ini')
	parser.add_argument('--tensorrt_config_file_path', help='config file of tensorrt', default='./config/tensorrt_config.ini')
	
	# Inference image & save path
	parser.add_argument('--inf_img_path', help='inference image path', default='./data/test_image/COCO_val2014_000000000761.jpg')
	parser.add_argument('--save_path', help='save path of inference image', default='./data/test_image')
		
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
	config_dict = get_config_dict([args.yolov3_config_file_path, args.train_config_file_path, args.tensorrt_config_file_path])
	
	if args.mode == 'yolov3-detection-img':
		
		# Read image
		img = np.array(pilimg.open(args.inf_img_path))
		print('Size of read image:', img.shape)
	
		# Yolov3 inference
		yolov3_image_detection_processor = Yolov3_Image_Dection_Processor(config_dict, device)
		box_list, class_list, conf_list, num_objects = yolov3_image_detection_processor.predict(img)
		
		print('-'*30)
		print('Object box list:', box_list)
		print('Class name list:', class_list)
		print('Conf score list:', conf_list)
		print('-'*30)
		
		# Draw objects
		draw_boxes(img, box_list, class_list, args.inf_img_path, args.save_path)
		
	elif args.mode == 'yolov3-train':
		yolov3_train_processor = Yolov3_Train_Processor(config_dict, device, random_seed_num)	
		yolov3_train_processor.train_eval()	
	
if __name__ == '__main__':
	main()
