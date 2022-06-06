
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
nvidia-docker run 
```

How to use
=============

#### - Build Yolov3 def cfg
```
./create_model_def.sh {class_num} {cfg_name}
```

#### - Build TensorRT engine
- Params: refer to parse_args()
```
python yolov3_convert_onnx_tensorrt.py --yolov3_config_file_path './config/yolov3_config.ini' --tensorrt_config_file_path './config/tensorrt_config.ini'
```

Config files of Yolov3 Train/Inference or 
=============
#### - Refer to config folder
- yolov3_config.ini: yolov3 model parameters
- train_config.ini: yolov3 train parameters
- tensorrt_config.ini: yolov3 tensorrt parameters

Dataset
=============

#### - Download COCO dataset
```
./get_coco_dataset.sh
```

#### - Build Data json files
- Building data json for optimizing yolov3
- In train process, read builded data json file and get train data
- Params: refer to parse_args()
```
python yolov3_convert_onnx_tensorrt.py --target 'coco2014' --data_folder_path './data/train_data/coco' --save_folder_path './data/data_json/coco'
```

#### - Format of data json files
- parsing_data_dic['class_format'] = type of class ('name' or 'id')
- parsing_data_dic['label_scale'] = scale of label ('absolute' or 'relative')
- parsing_data_dic['image_list'] = [{'id'-image id, 'image_file_path'-image file path}, ...]
- parsing_data_dic['object_boxes_list'] = [{'image_id'-image id, 'object_box_num'-number of the object per image, 'object_box_id_list'-[object box id, ...], 'object_name_list'-[object name, ...], 'object_box_list'-[[center x, center y, box_width, box_height], ...], 'object_box_size_list'-[object box size, ...], }, ...]
- parsing_data_dic['image_num'] = number of the image
- parsing_data_dic['object_boxes_num'] = [number of the total objects, ...]

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
