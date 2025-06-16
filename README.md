Contents
=============

#### - [Ultralytics Yolo11](https://docs.ultralytics.com/ko/models/yolo11/)

#### - [Ultralytics Yolo11 Solutions](https://docs.ultralytics.com/ko/solutions/)

#### - [Yolov11 Github](https://github.com/ultralytics/ultralytics)

#### - [Ultralytics Yolo12](https://docs.ultralytics.com/ko/models/yolo12)

#### - [Yolov12 Github](https://github.com/sunsmarterjie/yolov12)


Docker Environments
=============

#### - Build docker environment

``` 
sudo docker pull qbxlvnf11docker/human-vision-package:v2
```

#### - Run docker environment

```
sudo docker run -it --gpus all --name vision_package_env \
--shm-size=64G -p {port}:{port} -e GRANT_SUDO=yes --user root \
-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
-v {root_path}:/workspace/vision_pack \
-w /workspace/vision_pack qbxlvnf11docker/human-vision-package:v2 bash
```


Structures of Project Folders
=============

#### - CrowdHuman

  - 'datasets/CrowdHuman'
    - Original dataset folder of CrowdHuman

  - 'datasets/CrowdHuman_coco_format'
    - Preprocessing dataset folder of CrowdHuman for Ultralytics train format
    - Run 'python datasets/CrowdHuman_convertor.py'

#### - Safety Helmet Dataset

  - 'datasets/Safety_Helmet_Detection_with_Extended_Labels'
    - Original dataset folder of Safety Helmet Dataset

  - 'datasets/Safety_Helmet_Detection_with_Extended_Labels_coco_format'
    - Preprocessing dataset folder of Safety Helmet Dataset for Ultralytics train format
    - Run 'python datasets/CrowdHuman_convertor.py'

#### - Multi Dataset

  - 'datasets/multi_dataset'
    - preprocessing dataset folder of multi dataset for Ultralytics train format
    - Building it by concataneting many preprocessing dataset

#### - Custom Dataset

  - 'datasets/custom_labeling'
    - Original dataset folder of custom dataset with Anylabeling

  - 'datasets/custom_labeling_coco_format'
    - preprocessing dataset folder of custom dataset for Ultralytics train format
    - Run 'python datasets/auto_label_convertor.py'

        
```
${CODE_ROOT}
            |   |-- train_detector.py
            |   |-- demo.py
            |   |-- ...
            |   |-- datasets
            |   |   |   |-- CrowdHuman
            |   |   |   |   |   |-- CrowdHuman_train
            |   |   |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.jpg
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |-- CrowdHuman_val
            |   |   |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |   |   |-- 273271,1b9330008da38cd6.jpg
            |   |   |   |   |   |   |   |   |   |-- ...  
            |   |   |   |   |   |-- annotation_train.odgt
            |   |   |   |   |   |-- annotation_val.odgt
            |   |   |   |   |   |-- ...
            |   |   |   |-- CrowdHuman_coco_format
            |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |-- train
            |   |   |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.jpg
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |   |   |-- val
            |   |   |   |   |   |   |   |   |   |-- 273271,1b9330008da38cd6.jpg
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |-- labels
            |   |   |   |   |   |   |   |-- train
            |   |   |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.txt
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |   |   |-- val
            |   |   |   |   |   |   |   |   |   |-- 273271,1b9330008da38cd6.txt
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |-- Safety_Helmet_Detection_with_Extended_Labels
            |   |   |   |   |   |-- Images
            |   |   |   |   |   |   |   |-- hard_hat_workers0.png
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |-- Annotations
            |   |   |   |   |   |   |   |-- hard_hat_workers0.xml
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |-- Safety_Helmet_Detection_with_Extended_Labels_coco_format
            |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.jpg
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |-- labels
            |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.txt
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |-- multi_dataset
            |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.jpg
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |   |   |-- labels
            |   |   |   |   |   |   |   |-- 273271,1a0d6000b9e1f5b7.txt
            |   |   |   |   |   |   |   |-- ...
            |   |   |   |-- custom_labeling
            |   |   |   |   |   |-- custom_1.png
            |   |   |   |   |   |-- custom_1.json
            |   |   |   |   |   |-- ...
            |   |   |   |-- custom_labeling_coco_format
            |   |   |   |   |   |-- images
            |   |   |   |   |   |   |   |-- train
            |   |   |   |   |   |   |   |   |   |--  custom_1.png
            |   |   |   |   |   |   |   |   |   |--  ...
            |   |   |   |   |   |-- labels
            |   |   |   |   |   |   |   |-- train
            |   |   |   |   |   |   |   |   |   |--  custom_1.json
            |   |   |   |   |   |   |   |   |   |--  ...
            |   |   |   |-- ...
```


Build & Preprocssing Dataset
=============

#### - Ultralytics Settings

```
nano /root/.config/Ultralytics/settings.json
```

#### - CrowdHuman Dataset Preprocessing

  - Convert to COCO Format to train model
  - Class setting: {"person": 0, "head": 80} (Refer to Line 11)

```
python datasets/CrowdHuman_convertor.py 
```

#### - Safety Helmet Dataset Preprocessing

  - Convert to COCO Format to train model
  - Class setting: {"person_no_helmet": 0, "person_with_helmet": 0, "head": 80, "head_with_helmet": 81, "helmet": 82} (Refer to Line 12)

```
python datasets/safety_helmet_detection_dataset_convertor.py
```

#### - Build Nulti Dataset

  - Concatanete Preprocessing CrowdHuman Dataset folder and Preprocessing Safety Helmet Dataset folder

#### - Custom Dataset Labeling

  - Using Anylabeling labeling tools: [Anylabeling](https://github.com/vietanhdev/anylabeling)
  - How to use Anylabeling: [How to use Anylabeling](https://github.com/qbxlvnf11/SAM2-based-semi-auto-labeling)

#### - Build Custom Dataset

  - Convert to COCO Format to train model
  - Class setting: {"person_no_helmet": 0, "person_with_helmet": 0, "head": 80, "head_with_helmet": 81, "helmet": 82} (Refer to Line 12)

```
python datasets/auto_label_convertor.py
```


Run YOLO 11 & 12 Model
=============
   
#### - Pre-train weights: "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt", ...

#### - Fine-Tuning 
  - yolo_12_x using CrowdHuman Dataset: 'cfg/train/fine_tune_yolo12_x_crowd_human.yaml'
  - yolo_12_x using Safety Helmet Dataset: 'cfg/train/fine_tune_yolo12_x_safety_helmat.yaml'
  - yolo_12_x using Multi Dataset (Safety Helmet Dataset + CrowdHuman Dataset): 'cfg/train/fine_tune_yolo12_x_human_dataset.yaml'
  - yolo_12_x using Custom Dataset: 'cfg/train/fine_tune_yolo12_x_custom_dataset.yaml'
  - yolo_11_x using CrowdHuman Dataset: 'cfg/train/fine_tune_yolo11_x_crowd_human.yaml'
  - yolo_11_x using Safety Helmet Dataset: 'cfg/train/fine_tune_yolo11_x_safety_helmat.yaml'
  - yolo_11_x using Multi Dataset (Safety Helmet Dataset + CrowdHuman Dataset): 'cfg/train/fine_tune_yolo11_x_human_dataset.yaml'
  - yolo_11_x using Custom Dataset: 'cfg/train/fine_tune_yolo11_x_custom_dataset.yaml'

```
python fine_tuning.py --config {config_path}
```
   
#### - Inference
  - 'demo_yolo11.yaml'

```
python demo.py --config {config_path}
```


Author
=============

#### - [LinkedIn](https://www.linkedin.com/in/taeyong-kong-016bb2154)

#### - [Blog](https://blog.naver.com/qbxlvnf11)

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com


