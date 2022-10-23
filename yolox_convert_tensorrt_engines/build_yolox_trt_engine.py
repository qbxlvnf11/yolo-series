import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from PIL import Image
import cv2
import time
from loguru import logger
from configparser import ConfigParser
import sys

# Torch
import torch
from torch import nn
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

sys.path.append(".")

from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
from detection_util import load_classes
from yolox.data.data_augment import ValTransform

# ONNX: pip install onnx, onnxruntime
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

import tensorrt as trt
from torch2trt import torch2trt
from cuda import cuda 
import pycuda.autoinit

TRT_LOGGER = trt.Logger()

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Pytorch models to ONNX')
   
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')

    # Config file path
    parser.add_argument('--yolox_config_file_path', help='config file of yolox', default='./config/yolox_config.ini')
    
    # Sample image
    parser.add_argument('--img_size', help='image size',
        default=[3, 640, 640])
    parser.add_argument('--sample_image_path', help='sample image path',
        default='./datasets/test_image/1066405,2ac2400079a6d80f.jpg')

    # TensorRT engine params
    parser.add_argument('--tensorrt_engine_path',  help='tensorrt engine path',
        default='./yolox_convert_tensorrt_engines/yolox_tensorrt_engine.engine')
    parser.add_argument('--engine_workspace', type=int, help='workspace of engine', 
    	default=32)
    parser.add_argument('--engine_precision', help='precision of TensorRT engine', choices=['FP32', 'FP16'], 
    	default='FP16')
    parser.add_argument('--max_engine_batch_size', type=int, help='max batch size', 
    	default=1)
    	        
    args = parser.parse_args()

    return args
    
def get_transform(img_size):
    options = []
    options.append(transforms.Resize((img_size[1], img_size[2])))
    options.append(transforms.ToTensor())
    #options.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(options)
    return transform

def read_config(path):
	config = ConfigParser()
	config.read(path, encoding='utf-8') 
	
	return config

def get_config_dict(config_path_list):
	
	config_dict = {}
	
	for config_path in config_path_list:
		config_dict[config_path.split('/')[-1].split('.')[0]] = read_config(config_path)
	
	return config_dict

def load_model(model, weight_path, device):
    
    print('Detection model load ...')
    print('Load weights from {}.'.format(weight_path))
    
    checkpoint = torch.load(weight_path, map_location=device)
    
    # Load pretrained model
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    model.head.decode_in_inference = False
    
    return model

def trt_inference(engine, context, data):  
    
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    print('nInput:', nInput)
    print('nOutput:', nOutput)
    
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput,nInput+nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
        
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)
    
    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)
        
    for b in bufferD:
        cuda.cuMemFree(b)  
    
    return bufferH
    
def main():
    args = parse_args()

    # Config
    config_dict = get_config_dict([args.yolox_config_file_path])
    yolox_config = config_dict['yolox_config']

    # Class list
    object_class_names_list_path = yolox_config.get('class_names', 'object_class_names_list_path', raw=True)
    object_class_names_list = load_classes(object_class_names_list_path)
    class_num = len(object_class_names_list)
    print('Object class names list:', object_class_names_list)
    print('Class num:', class_num)

    # Threshold
    object_confidence_threshold = yolox_config.getfloat('threshold', 'object_confidence_threshold')
    nms_thres = yolox_config.getfloat('threshold', 'nms_thres')

    ## Sample image
    # Get img
    img = cv2.imread(args.sample_image_path, cv2.IMREAD_COLOR)
 
    # Preprocessing
    preproc = ValTransform(legacy=False)
    resize_width_height = args.img_size[1]
    img_resize, _ = preproc(img, None, (resize_width_height, resize_width_height))
    img_resize = torch.from_numpy(img_resize).unsqueeze(0)
    img_resize = img_resize.float().to(args.device)
    print('Inference image size:', img_resize.shape)
    
    ## YoloX Pytorch model	
    # Get model
    model_name = yolox_config.get('yolox', 'model_name')
    print('Model name:', model_name)
    exp = get_exp(None, None, object_class_names_list, resize_width_height, None, model_name)
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, (resize_width_height, resize_width_height))))
    
    # Load weight
    weight_path = yolox_config.get('yolox', 'weight_path', raw=True)
    model = load_model(model, weight_path, args.device)
    print('weight path:', weight_path)
    
    ## Pytorch results
    # Inf
    torch_start_time = time.time()
    pytorch_result = model(img_resize)
    torch_end_time = time.time()
    
    # Postprocess
    pytorch_result = postprocess(pytorch_result, class_num, object_confidence_threshold, nms_thres)[0]
    print('Shape of postprocessing pytorch result:', pytorch_result.shape)

    ## Pytorch to TensorRT engine
    # Set FP16
    if args.engine_precision == 'FP16':
    	fp16_mode = True
    else:
    	fp16_mode = False
    	
    # Build engine
    if not os.path.exists(args.tensorrt_engine_path):
	    yolox_trt_engine = torch2trt(
		model,
		[img_resize],
		fp16_mode=fp16_mode,
		log_level=trt.Logger.INFO,
		max_workspace_size=(1 << args.engine_workspace),	
		max_batch_size=args.max_engine_batch_size,
	    )
	    print('Complete to convert YOLOX Pytorch to TensorRT')
	    
	    # Write engine
	    with open(args.tensorrt_engine_path, "wb") as f:
	    	f.write(yolox_trt_engine.engine.serialize())
    
    # Read engine
    with open(args.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
        yolox_trt_engine = runtime.deserialize_cuda_engine(f.read())    
    
    ## TensorRT inference
    # Context
    context = yolox_trt_engine.create_execution_context()
    context.set_binding_shape(0, (1, 3, resize_width_height, resize_width_height))
    
    # Inf
    trt_start_time = time.time()
    trt_outputs = trt_inference(yolox_trt_engine, context, img_resize.cpu().numpy())
    trt_end_time = time.time()
    
    # Postprocess
    trt_outputs = np.array(trt_outputs[1])
    trt_outputs = torch.tensor(trt_outputs).to(args.device)
    trt_outputs = postprocess(trt_outputs, class_num, object_confidence_threshold, nms_thres)[0]
    print('Shape of postprocessing trt result:', trt_outputs.shape)

    ## Comparision output of TensorRT and output of onnx model
    # Time Efficiency & output
    print('--pytorch--')
    print(np.argmax(pytorch_result.detach().cpu().numpy(), axis=1))
    print('Time:', torch_end_time - torch_start_time)
    
    print('--tensorrt--')
    print(np.argmax(trt_outputs.detach().cpu().numpy(), axis=1))
    print('Time:', trt_end_time - trt_start_time)
    
if __name__ == '__main__':
    main()
