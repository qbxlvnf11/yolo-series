import argparse
from configparser import ConfigParser

from yolov3_convert_tensorrt_engines.yolov3_to_onnx import yolov3_to_onnx
from yolov3_convert_tensorrt_engines.onnx_to_tensorrt import onnx_to_tensorrt

def parse_args():
	parser = argparse.ArgumentParser(
		description='Yolo-v3 model train/inference')
	
	# CUDA or not
	parser.add_argument('--device', help='device of yolov3 (cpu, cuda:0, ...)', default='cuda:0')
	
	# Config file path
	parser.add_argument('--yolov3_config_file_path', help='config file of yolov3', default='./config/yolov3_config.ini')
	parser.add_argument('--tensorrt_config_file_path', help='config file of tensorrt', default='./config/tensorrt_config.ini')
	
	# Inference image & save path
	parser.add_argument('--inf_img_path', help='inference image path', default='./test_image/COCO_val2014_000000000761.jpg')
	parser.add_argument('--save_path', help='save path of inference image', default='./test_image')
		
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
	
	# Config dict
	config_dict = get_config_dict([args.yolov3_config_file_path, args.tensorrt_config_file_path])
	
	yolov3_to_onnx(config_dict)
	
	onnx_to_tensorrt(config_dict, args.inf_img_path)
		
if __name__ == '__main__':
	main()
