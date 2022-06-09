#!/usr/bin/env python2
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import cv2
import time
import numpy as np
from PIL import Image

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()

from yolov3_convert_tensorrt_engines.convert_yolov3_tensorrt_util import PreprocessYOLO, PostprocessYOLO

def load_classes(path):
	"""
	Loads class labels at 'path'
	"""
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

def get_engine(onnx_file_path, batch_size, resize_width_height, engine_file_path=""):
 
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = batch_size
            builder.fp16_mode = True
            
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [batch_size, 3, resize_width_height, resize_width_height]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def onnx_to_tensorrt(config_dict, inf_img_path):
    """Create a TensorRT engine for ONNX-based YOLOv3 and run inference."""

    ## Config
    yolov3_config = config_dict['yolov3_config']
    tensorrt_config = config_dict['tensorrt_config']
    
    ## Params
    yolov3_channels = yolov3_config.getint('yolov3', 'yolov3_channels')
    resize_width_height = yolov3_config.getint('yolov3', 'resize_width_height')
    save_onnx_model_path = tensorrt_config.get('onnx_tensorrt_engine_path', 'save_onnx_model_path', raw=True)
    save_tensorrt_engine_path = tensorrt_config.get('onnx_tensorrt_engine_path', 'save_tensorrt_engine_path', raw=True)   
    batch_size = 1
    object_class_names_list_path = yolov3_config.get('class_names', 'object_class_names_list_path', raw=True)
    object_class_names_list = load_classes(object_class_names_list_path)

    class_num = len(object_class_names_list)
    print('class_num:', class_num)
    
    # Read image
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (resize_width_height, resize_width_height)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    raw_img, img = preprocessor.process(inf_img_path)
    print('Input img:', img.shape)
    shape_orig_WH = raw_img.size
    print('Input raw img:', shape_orig_WH)

    # Output shapes expected by the post-processor
    output_shapes = [(batch_size, (class_num + 5) * 3, resize_width_height // 32, resize_width_height // 32),
                     (batch_size, (class_num + 5) * 3, resize_width_height // 16, resize_width_height // 16),
                     (batch_size, (class_num + 5) * 3, resize_width_height // 8,  resize_width_height // 8)]
    print('output_shapes:', output_shapes)
                     
    # Do inference with TensorRT
    with get_engine(save_onnx_model_path, batch_size, resize_width_height, save_tensorrt_engine_path) as engine, engine.create_execution_context() as context:
        
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        
        print('Running inference on image {}...'.format(inf_img_path))
        
        trt_start_time = time.time()
      
        # Set host input to the image.
        inputs[0].host = img
        
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) #, batch_size=batch_size)
            
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        
        trt_end_time = time.time()

    ## ONNX inference
    onnx_model = onnx.load(save_onnx_model_path)
    
    # Check onnx 
    onnx.checker.check_model(onnx_model)
    
    sess = rt.InferenceSession(save_onnx_model_path)
    
    sess_input = sess.get_inputs()[0].name
    sess_output1 = sess.get_outputs()[0].name
    sess_output2 = sess.get_outputs()[1].name
    sess_output3 = sess.get_outputs()[2].name

    onnx_start_time = time.time()
        
    onnx_results = sess.run([sess_output1, sess_output2, sess_output3], {sess_input: img.astype(np.float32)})
    
    onnx_end_time = time.time()

    # Time Efficiency & output
    print('--onnx--')
    print('onnx_result[0]:', onnx_results[0].shape)
    print('onnx_result[1]:', onnx_results[1].shape)
    print('onnx_result[2]:', onnx_results[2].shape)
    print(np.argmax(onnx_results[0], axis=1))
    print('time: ', onnx_end_time - onnx_start_time)
    
    print('--tensorrt--')
    print('trt_outputs[0]:', trt_outputs[0].shape)
    print('trt_outputs[1]:', trt_outputs[1].shape)
    print('trt_outputs[2]:', trt_outputs[2].shape)
    print(np.argmax(trt_outputs[0], axis=1))
    print('time: ', trt_end_time - trt_start_time)

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW,
                          "class_num": class_num}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    print('boxes:', boxes)
    print('classes:', classes)
    print('scores:', scores)

if __name__ == '__main__':
    main()
