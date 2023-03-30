Description
=============

#### - Yolo object detector series of various version (refer to each branch or tag)

#### - Contributions of Yolov7
  - Several trainable bag-of-freebies methods are designed to improve the performance of real-time object detection without increasing inference cost.
  - For the development of detection methods, the following two issues were discovered and solutions were proposed. (1) How can the original module be replaced with a re-parameterized module? (2) How dynamic label assignment strategy deals with assignment to different output layers.
  - "extend" and "compound scaling" methods are proposed so that real-time object detectors can efficiently utilize parameters and computations.
  - The proposed Yolov7 maintained high performance despite reducing parameters by about 40% and computation by about 50%.

#### - Summary of Yolov7 Mechanism
  - Blog post link: https://blog.naver.com/qbxlvnf11/223056418459
  - Characteristic
    - E-ELAN Architecture

    <img src="https://user-images.githubusercontent.com/52263269/228470825-01baf4f0-c06f-480b-8e64-003f99ab17f4.png" width="90%"></img>

    - Model Scaling for Concatenation-Based Model Architecture

    <img src="https://user-images.githubusercontent.com/52263269/228469230-30ff0446-7d33-4cb8-8511-7a466a16b890.png" width="90%"></img>

    - Planned re-parameterized convolution

    <img src="https://user-images.githubusercontent.com/52263269/228472239-500dc738-de9d-433a-a554-d723326d7794.png" width="50%"></img>

    - Coarse for auxiliary and fine for lead loss

    <img src="https://user-images.githubusercontent.com/52263269/228472541-7494916c-7743-4c5a-888e-aebcb4bfb99c.png" width="90%"></img>


Contents
=============

#### - Modify Yolov7 code in [Yolov7 official repository](https://github.com/WongKinYiu/yolov7) to make object detector optimize for human detection (jointly learning of CrowdHuman, Safety Helmet Dataset)

#### - Yolov7 Train/Fine-tune/Validate/Inference
  - Train & Fine-tune Yolov7 model
    - Fine-tune with custom human detection dataset: jointly learning of CrowdHuman, Safety Helmet Dataset (refet to cache_labels method in '/utils/dataset.py')
    - Caution! OTA (Optimal Transport Assignment for Object Detection) loss likeyly to cause GPU memory overflow when maximum length of label is very long (e.g. 782 in CrowdHuman)
    - This problem can be addressed by modifying the parameters of the configuration file to limit the maximum length of label or not use OTA loss.
      - Limiting the maximum length of label: e.g. set 'cut_max_len' parameter as 200 in human_custom.yaml
      - Not use OTA loss: e.g. set 'loss_ota' parameter as 1 in hyp.scratch.human_custom.yaml

  - Test & Inference Yolov7 model
    - Test: Confusion Matrix, F1/PR/P/R Curve etc.
    - Inference: Detect objects in image
    
  <img src="https://user-images.githubusercontent.com/52263269/228701002-7795546e-caa8-4667-9409-a1ec6e161a58.jpg" width="45%"></img> 
  <img src="https://user-images.githubusercontent.com/52263269/228700447-7c625fa1-09ba-4982-a233-c0631c31d25b.jpg" width="45%"></img>

  <img src="https://user-images.githubusercontent.com/52263269/228702551-36043d61-931d-4322-ac20-112d7f6cf3ad.jpg" width="45%"></img> 
  <img src="https://user-images.githubusercontent.com/52263269/228702485-8784d840-e686-4492-9c08-a7207500ced3.jpg" width="45%"></img>

#### - Convert & Inference Yolov7 TensorRT Engine
- Convert Yolov7 Pytorch weigths to TensorRT engine: FP16, INT8 calibration
- Faster inference of Yolov7 TensorRT engine

#### - Config files
- Build config for joint learning of two human dataset


Structures of Project Folders
=============

```
${CODE_ROOT}
            |   |-- train.py
            |   |-- ...
${DATA_ROOT}
            |   |-- train_total_data_path_list.txt    
            |   |-- valid_total_data_path_list.txt
            |   |-- CrowdHuman
            |   |   |   |-- CrowdHuman_train01
            |   |   |   |-- CrowdHuman_train02
            |   |   |   |-- CrowdHuman_train03
            |   |   |   |-- CrowdHuman_val
            |   |   |   |-- CrowdHuman_test
            |   |   |   |-- annotation_train.odgt
            |   |   |   |-- annotation_val.odgt
            |   |-- Safety_Helmet_Detection_with_Extended_Labels
            |   |   |   |-- Images
            |   |   |   |-- Annotations
            |   |-- COCO2017
            |   |   |   |-- images
            |   |   |   |-- labels
            |   |   |   |-- train2017.txt
            |   |   |   |-- val2017.txt
            |   |   |   |-- test-dev2017.txt
```


Custom Human Detection Dataset
=============

#### - Path of data_path_list.txt
  - Train: './data/train_total_data_path_list.txt'
  - Valid: './data/valid_total_data_path_list.txt'

#### - Crowd Human Dataset

https://www.crowdhuman.org/

https://www.crowdhuman.org/download.html

#### - Safety Helmet detection with Extended Labels (SHEL) Dataset

https://data.mendeley.com/datasets/9rcv8mm682/2


Download Weights & TensorRT Engine
=============

#### - Download COCO pretrained weights of Yolov7

https://github.com/WongKinYiu/yolov7

#### - Download fine-tuned weights and TensorRT engine of Yolov7
  - Password: 1234
  
http://naver.me/5bdUjMvg


Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/yolov7_tensorrt
```

#### - Run docker environment

```
nvidia-docker run -it --gpus all --name yolov7_tensorrt --shm-size=64G -p 8844:8844 -e GRANT_SUDO=yes --user root -v {data_folder}:/workspace/data -v {yolov7_folder}:/workspace/yolov7 -w /workspace/yolov7 qbxlvnf11docker/yolov7_tensorrt bash
```


How to use
=============

#### - Train Yolov7: Pre-Train or Fine-Tuning
  - COCO pretrained: P5 & P6

  ```
  python train.py --workers 8 --device 0 --batch-size 16 --data data/coco_custom.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7-coco-custom --hyp data/hyp.scratch.custom.yaml --epochs 300
  ```
  
  ```
  python train_aux.py --workers 8 --device 0 --batch-size 8 --data data/coco_custom.yaml --img 640 640 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6-coco-custom --hyp data/hyp.scratch.custom.yaml --epochs 300
  ```

  - COCO pretrained + Custom Human Detection Dataset Fine-Tune: P5 & P6

  ```
  python train.py --workers 8 --device 0 --batch-size 16 --data data/human_custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights ./weights/yolov7.pt --name yolov7-human-custom --hyp data/hyp.scratch.human_custom.yaml --epochs 100
  ```

  ```
  python train_aux.py --workers 8 --device 0 --batch-size 2 --data data/human_w6_custom.yaml --img 640 640 --cfg cfg/training/yolov7-w6-custom.yaml --weights ./weights/yolov7-w6.pt --name yolov7-w6-human-custom --hyp data/hyp.scratch.human_custom.yaml --epochs 100
  ```

#### - Test Yolov7: Confusion Matrix, F1/PR/P/R Curve etc.
  - COCO pretrained Weights

  ```
  python test.py --data data/coco_custom.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights ./weights/yolov7.pt --name yolov7_coco_val --no-trace
  ```
  
  - COCO pretrained + Custom Human Detection Dataset Fine-Tune Weights

  ```
  python test.py --data data/human_custom.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights ./weights/yolov7_human.pt --name yolov7_human_val --no-trace
  ```

#### - Building Yolov7 ONNX
  - For inference: add '--max-wh 640'
  - COCO pretrained Weights

  ```
  python export_onnx.py --weights ./weights/yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
  ```
  
  - COCO pretrained + Custom Human Detection Dataset Fine-Tune Weights

  ```
  python export_onnx.py --weights ./weights/yolov7_human.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
  ```

#### - Building Yolov7 FP16 TensorRT Engines (ONNX to TensorRT)
  - Clone tensorrt-python reposit
  
  ```
  git clone https://github.com/Linaom1214/tensorrt-python.git
  ```
  
  - COCO pretrained ONNX

  ```
  python ./tensorrt-python/export_trt.py -o ./weights/yolov7.onnx -e ./weights/yolov7_FP16.trt -p fp16
  ```
  
  - COCO pretrained + Custom Human Detection Dataset Fine-Tune ONNX
  
  ```
  python ./tensorrt-python/export_trt.py -o ./weights/yolov7_human.onnx -e ./weights/yolov7_human_FP16.trt -p fp16
  ```

#### - Building Yolov7 INT8 Calibration TensorRT Engines (ONNX to TensorRT)
  - Clone tensorrt-python reposit
  
  ```
  git clone https://github.com/Linaom1214/tensorrt-python.git
  ```
  
  - COCO pretrained ONNX

  ```
  python ./tensorrt-python/export_trt.py -o ./weights/yolov7.onnx -e ./weights/yolov7_INT8.trt -p int8 --calib_input ./samples/images --calib_cache ./weights/calibration.cache
  ```
  
  - COCO pretrained + Custom Human Detection Dataset Fine-Tune ONNX
  
  ```
  python ./tensorrt-python/export_trt.py -o ./weights/yolov7_human.onnx -e ./weights/yolov7_human_INT8.trt -p int8 --calib_input ./samples/images --calib_cache ./weights/calibration.cache
  ```

#### - Pytorch Inference: detecting object with pretrained Yolov7
  - COCO pretrained Weights

  ```
  python detect.py --weights ./weights/yolov7.pt --conf 0.25 --img-size 640 --source samples/images/horses.jpg --no-trace
  ```
  
  - COCO pretrained + Custom Human Detection Dataset Fine-Tune Weights
  
  ```
  python detect.py --weights ./weights/yolov7_human.pt --conf 0.4 --img-size 640 --source samples/images/1066405,2bfbf000c47880b7.jpg --no-trace
  ```

#### - ONNX Inference: detecting object with Yolov7 ONNX
  - COCO pretrained ONNX
  
  ```
  python detect.py --onnx-inf --onnx-path ./weights/yolov7.onnx --weights ./weights/yolov7.pt --conf 0.25 --img-size 640 --source samples/images/horses.jpg --no-trace
  ```

  - COCO pretrained + Custom Human Detection Dataset Fine-Tune ONNX

  ```
  python detect.py --onnx-inf --onnx-path ./weights/yolov7_human.onnx --weights ./weights/yolov7_human.pt --conf 0.4 --img-size 640 --source samples/images/1066405,2bfbf000c47880b7.jpg --no-trace
  ```

#### - FP16 TRT Inference: detecting object with Yolov7 FP16 TensorRT engine
  - COCO pretrained TRT
  
  ```
  python detect.py --trt-inf --trt-engine-path ./weights/yolov7_FP16.trt --weights ./weights/yolov7.pt --conf 0.25 --img-size 640 --source samples/images/horses.jpg --no-trace
  ```

  - COCO pretrained + Custom Human Detection Dataset Fine-Tune TRT

  ```
  python detect.py --trt-inf --trt-engine-path ./weights/yolov7_human_FP16.trt --weights ./weights/yolov7_human.pt --conf 0.25 --img-size 640 --source samples/images/1066405,2bfbf000c47880b7.jpg --no-trace
  ```

#### - INT8 TRT Inference: detecting object with Yolov7 INT8 Calibration TensorRT engine
  - COCO pretrained TRT
  
  ```
  python detect.py --trt-inf --trt-engine-path ./weights/yolov7_INT8.trt --weights ./weights/yolov7.pt --conf 0.25 --img-size 640 --source samples/images/horses.jpg --no-trace
  ```

  - COCO pretrained + Custom Human Detection Dataset Fine-Tune TRT

  ```
  python detect.py --trt-inf --trt-engine-path ./weights/yolov7_human_INT8.trt --weights ./weights/yolov7_human.pt --conf 0.25 --img-size 640 --source samples/images/1066405,2bfbf000c47880b7.jpg --no-trace
  ```


References
=============

#### - Yolov7 paper
```
@article{Yolov7,
  title={YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao Institute of Information Science, Academia Sinica, Taiwan},
  journal = {arXiv},
  year={2022}
}
```

#### - Yolov7 Pytorch & TensorRT

https://github.com/WongKinYiu/yolov7

#### - torch2trt

https://github.com/NVIDIA-AI-IOT/torch2trt


Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
