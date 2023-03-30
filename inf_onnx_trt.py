import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from collections import OrderedDict,namedtuple
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def inference_trt(im, bindings, binding_addrs, context, device):

    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

    # images = bindings['images'].data
    # output = bindings['output'].data
    # print('shape:', images.shape, output.shape)
    # print(output[0])

    nums = bindings['num_dets'].data
    boxes = bindings['det_boxes'].data
    scores = bindings['det_scores'].data
    classes = bindings['det_classes'].data

    boxes = boxes[0,:nums[0][0]]
    scores = scores[0,:nums[0][0]]
    classes = classes[0,:nums[0][0]]
    print('shape:', nums.shape, boxes.shape, scores.shape, classes.shape)

    output = torch.zeros((nums[0][0], 7)).to(device)
    for a in range(nums[0][0]):
        output[a][0] = 0
        output[a][1:5] = boxes[a]
        output[a][5] = classes[a]
        output[a][6] = scores[a]

    return output

def get_tensorrt_engine(trt_engine_path, device, img_size=640):

    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")

    with open(trt_engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    
    bindings = OrderedDict()

    print('-'*20)
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        print('binding name:', name)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    print('-'*20)

    # Warmup for 20 times
    for _ in range(20):
        tmp = torch.randn(1, 3, img_size, img_size).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

    return bindings, binding_addrs, context

def get_onnx_session(onnx_path):

    print('-'*20)
    print('onnx path:', onnx_path)
    
    opts = ort.SessionOptions()
    opts.enable_profiling = True

    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, opts, providers=providers)

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    print('onnx outname:', outname)
    print('onnx inname:', inname)
    print('-'*20)

    return session, inname, outname

def inference_onnx(session, img, inname, outname):

    inp = {inname[0]:img}

    outputs = session.run(outname, inp)[0]

    return outputs