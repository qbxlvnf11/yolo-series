
Description
=============

#### - Yolov3

Contents
=============

#### - Yolov3 Train
#### - Yolov3 Inference
#### - Converting Yolov3 Pytorch Weigths to TensorRT Engines 

Yolov3 Run Environments with TensorRT 7.2.2 & Pytorch
=============

#### - Docker pull
```
docker pull 
```

#### - Docker run
```
nvidia-docker pull 
```

How to use
=============

#### - Build Yolov3 def cfg
```
./create_model_def.sh {class_num} {cfg_name}
```

#### - Build TensorRT engine
```
yolov3_convert_onnx_tensorrt.py
```

Yolov3 Run Environments with TensorRT 7.2.2 & Pytorch
=============
#### - Parameters

References
=============

#### - Yolov3 Pytorch

https://github.com/eriklindernoren/PyTorch-YOLOv3

#### - Yolov3 TensorRT

https://github.com/linghu8812/YOLOv3-TensorRT


Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
