Description
=============

#### - Yolo object detector series of various version (refer to each branch)

#### - YoloX
  - You only look once (YOLO) is one of the the powerful and real-time 1-stage object detection systems
  - Structure: 'Input - Backbone - Neck - Dense Prediction (Head)'
    - Darknet53 backbone with SPP layer: extracting feature map
    - FPN nect
      - Multi-scale feature map
      - Detecting large objects from the feature map with small resolution and detecting small objects from the feature map with high resolution
    - Decoupled head
    
    <img src="https://user-images.githubusercontent.com/52263269/197380708-87811312-742e-45ae-922a-8ff7370deceb.png" width="50%"></img>

Contents
=============

#### - YoloX Train/inference
  - Train YoloX model
  - Fine-tune YoloX model
  - Detect objects in image
  
  <img src="https://user-images.githubusercontent.com/52263269/197380793-d6d9aa43-0c60-4582-8205-27e08e885ca7.jpg" width="30%"></img>
  <img src="https://user-images.githubusercontent.com/52263269/197380863-12d6792b-9e05-4009-ab47-00247c55ed51.png" width="30%"></img>

  <img src="https://user-images.githubusercontent.com/52263269/197380885-7d03b854-b98f-407f-8923-8d02a97c3091.jpg" width="30%"></img>
  <img src="https://user-images.githubusercontent.com/52263269/197380898-421853b0-1627-447b-94a2-666897373bb4.png" width="30%"></img>

#### - YoloX TensorRT Engine
- Convert YoloX Pytorch weigths to TensorRT engine
- Real-time inference with YoloX TensorRT engine

#### - Config files
- yolox_config.ini: yoloX model parameters
- train_config.ini: yoloX train parameters
- tensorrt_config.ini: yoloX tensorrt parameters

Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/yolox_tensorrt
```

#### - Run docker environment

```
nvidia-docker run -it -p 9000:9000 -e GRANT_SUDO=yes --user root --name yolox_tensorrt -v {folder}:/workspace -w /workspace qbxlvnf11docker/yolox_tensorrt bash
```

How to use
=============

#### - Train: training or fine-tuning YoloX
  - Params: refer to '../config/train_config.ini', '../config/yolox_config.ini' config files and parse_args() of 'main.py'

```
python main.py --mode yolox-train
```

#### - Inference: detecting object with pretrained YoloX
  - Params: refer to three config files and parse_args() of 'main.py'
  - TensorRT engine inference: set tensorrt_mode config of '../config/tensorrt_config.ini' as 'yes' after build YoloX TensorRT engines

```
python main.py --mode yolox-detection-img
```

#### - Building YoloX TensorRT Engines
  - Params: refer to '../config/yolox_config.ini' config file and parse_args() of '../yolox_convert_tensorrt_engines/build_yolox_trt_engine.py'

```
python yolox_convert_tensorrt_engines/build_yolox_trt_engine.py
```

Train with Custom Dataset
=============

#### - Build Dataset
  
1. Building Custom Dataset Class
  - Building dataset class py file in '../yolox/data/datasets'
  
2. Import Custom Dataset Class
  - Writing import code in '../yolox/data/datasets/__init__.py'
  - Writing import code in get_data_loader, get_eval_loader function of '../yolox/exp/yolox_base.py'
  
3. Define Custom Dataset Class
  - Defining dataset class in get_data_loader, get_eval_loader function of '../yolox/exp/yolox_base.py'

#### - Config Setting

1. Set follow configs of '../config/train_config.ini'
  - dataset_name: choosing custom dataset
  - data_dir: custom data folder path

2. Set follow config of '../config/yolox_config.ini'
  - object_class_names_list_path: class names file path
  - Add class names that does not exist in the existing class names list or change class names file

#### - Refer to 'CrowdHumanDataset' custom dataset class

Transfer Learning (Fine-Tune)
=============

#### - Three modes of Fine-Tune

1. Load weights of all layers
  - Set fine_tune_mode config of '../config/train_config.ini' as 'yes'

2. Load only weights of DarkNet
  - Set fine_tune_mode config of '../config/train_config.ini' as 'yes'
  - Set only_backbone_weight_load_mode config of '../config/train_config.ini' as 'yes'

3. Load weights of DarkNet and head's layers except for last layer
  - Set fine_tune_mode config of '../config/train_config.ini' as 'yes'
  - Set except_predict_layer_weight_load_mode config of '../config/train_config.ini' as 'yes'

Download Weights & TensorRT Engine
=============

#### - Download COCO pretrained weights of YoloX

https://github.com/Megvii-BaseDetection/YOLOX

#### - Download Crowd Human Dataset fine-tuned weights and TensorRT engine of YoloX
  - Password: 1234
  
http://naver.me/5SavC6wN

References
=============

#### - YoloX paper
```
@article{yolox,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun},
  journal = {arXiv},
  year={2021}
}
```

#### - YoloX Pytorch

https://github.com/Megvii-BaseDetection/YOLOX

#### - Yolox TensorRT engine

https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/trt.py

#### - torch2trt

https://github.com/NVIDIA-AI-IOT/torch2trt

#### - Inference TensorRT engine

https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt

#### - Crowd Human Dataset

https://www.crowdhuman.org/

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
