import time
import copy
import torch
import os
from datetime import datetime
import logging
import cv2

logger = logging.getLogger(__name__)

from ultralytics import YOLO
from inference.models.yolo_world.yolo_world import YOLOWorld
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

from utils.formats import xywh_to_xyxy


class Model():

    def __init__(self, config, classes=None, vis_folder=None):
        
        self.model_version = config['model_version']

        self.input_shape = (config['input_shape']['height'], config['input_shape']['width'])
        self.save_cropped_person = config['save_cropped_person']
        self.vis_folder = vis_folder

        self.classes = classes
        self.detection_confidence_thr = config['detection']['confidence_threshold']
        self.detection_iou_thr = config['detection']['detection_iou_thr']
        self.max_det = config['detection']['max_det']
        
        self.keypoints_working = config['pose_estimation']['working']
        self.pose_confidence_thr = config['pose_estimation']['confidence_threshold']
        self.tracking_confidence = config['pose_estimation']['tracking_confidence']

        if self.model_version == 'yolo_world/deep_sort/mp_pose':
            ## Detection
            self.detector = YOLOWorld(model_id=config['detection']['yolo_world_model_id'])
            self.detector.set_classes(self.classes)
            # logging.info(f'Load weights: {self.detector.load_weights}')
            # logging.info(f'Model: {self.detector.model}')

            ## Tracking
            self.tracker = DeepSort(max_age=config['tracking']['max_age'], \
                nms_max_overlap=config['tracking']['nms_max_overlap'], embedder_gpu=True)
            self.reset_seconds = config['fps'] * config['tracking']['reset_seconds']

            ## Pose Estimation
            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(static_image_mode=False, \
                model_complexity=1, \
                enable_segmentation=False, \
                min_detection_confidence=self.pose_confidence_thr, \
                min_tracking_confidence=self.tracking_confidence)

        elif self.model_version == "ultralytics/mp_pose":

            ## Detection
            self.detector = YOLO(config['detection']['ultralytics_fine_tuning_model_weights'])
            self.detector.info()
            self.class_dict = self.detector.names
            self.class_dict = {v: k for k, v in self.class_dict.items()}

            if self.classes is not None:
                new_list = []
                for class_name in self.classes:
                    if class_name in self.class_dict:
                        new_list.append(self.class_dict[class_name])
                self.classes = new_list

            logging.info(f'Class names: {self.class_dict}')

            ## Pose Estimation
            # self.pose_estimator = YOLO(config['pose_estimation']['ultralytics_pose_model_weights'])
            # self.pose_estimator.info()

            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(static_image_mode=False, \
                model_complexity=1, \
                enable_segmentation=False, \
                min_detection_confidence=self.pose_confidence_thr, \
                min_tracking_confidence=self.tracking_confidence)

        elif self.model_version == "ultralytics/ultralytics":

            ## Detection
            self.detector = YOLO(config['detection']['ultralytics_fine_tuning_model_weights'])
            self.detector.info()
            self.class_dict = self.detector.names
            self.class_dict = {v: k for k, v in self.class_dict.items()}

            if self.classes is not None:
                new_list = []
                for class_name in self.classes:
                    if class_name in self.class_dict:
                        new_list.append(self.class_dict[class_name])
                self.classes = new_list

            logging.info(f'Class names: {self.class_dict}')

            ## Pose Estimation
            self.pose_estimator = YOLO(config['pose_estimation']['ultralytics_pose_model_weights'])
            self.pose_estimator.info()

    def detect_object(self, frame):

        detections = self.detector.infer(frame, confidence=self.detection_confidence_thr)

        return detections

    def apply_tracking(self, frame, frame_id, detections):

        tracks = self.tracker.update_tracks(detections, frame=frame)

        if frame_id % self.reset_seconds == 0:
            self.tracker.tracks = []

        tracked_objects = []
        for i, track in enumerate(tracks):

            if not track.is_confirmed(): # or track.time_since_update > 1:
                continue
            
            tracked_objects.append({
                "track_id": track.track_id,
                "bbox": track.to_ltrb(),
                "class_id": track.det_class,
                # "confidence": detections[i][1]
            })

        return tracked_objects

    def extract_pose(self, frame):
        keypoints = self.pose_estimator.process(frame)

        return keypoints

    def inference(self, frame, frame_id=-1, video_name=None):
        
        if self.model_version == 'yolo_world/deep_sort/mp_pose':
            bbox_results = []
            
            ## Object detection
            start_time = datetime.now()
            detections = self.detect_object(frame).predictions
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            logging.info(f'Detection time: {elapsed_time}')
            # logging.info(f'Detections: {detections}')

            ## Tracking
            tracker_inputs = []
            keypoints_list = []

            for detection in detections:
                    
                x, y, width, height = int(detection.x), int(detection.y), int(detection.width), int(detection.height)
                x_min, y_min, x_max, y_max = xywh_to_xyxy(x, y, width, height)

                confidence = detection.confidence
                # class_name = detection.class_name
                class_id = detection.class_id

                if class_id == 0:
                    tracker_inputs.append([[x_min, y_min, width, height], confidence, class_id])

                    ## Keypoints
                    start_time = datetime.now()
                    cropped_frame = copy.deepcopy(frame)[int(y_min):int(y_max), int(x_min):int(x_max)]
                    keypoints = self.extract_pose(cropped_frame)
                    if keypoints.pose_landmarks:
                        for landmark in keypoints.pose_landmarks.landmark:
                            landmark.x = landmark.x * (x_max - x_min) / frame.shape[1] + x_min / frame.shape[1]
                            landmark.y = landmark.y * (y_max - y_min) / frame.shape[0] + y_min / frame.shape[0]
                    keypoints_list.append(keypoints)
                    end_time = datetime.now()
                    elapsed_time = (end_time - start_time).total_seconds()
                    logging.info(f'Keypoints estimation time per person: {elapsed_time}')     
                else:
                    bbox_results.append({"bbox":[x_min, y_min, x_max, y_max], "class_id":class_id})
            
            start_time = datetime.now()
            tracked_objects = self.apply_tracking(frame, frame_id, tracker_inputs)
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            logging.info(f'Tracking time: {elapsed_time}')
            # logging.info(f'Tracked objects: {tracked_objects}')

            for tracked_object in tracked_objects:
                bbox_results.append(tracked_object)
            # logging.info(f'Box results: {bbox_results}')

            return {'bbox':bbox_results, 'keypoints': keypoints_list, 'pose_connections':self.mp_pose.POSE_CONNECTIONS}
        
        elif self.model_version == "ultralytics/mp_pose":
            # detections = self.detector.predict(frame, save=False, classes=self.classes, \
            #     imgsz=self.input_shape, iou=self.detection_iou_thr, conf=self.detection_confidence_thr, \
            #     max_det=self.max_det)

            detections = self.detector.track(frame, save=False, classes=self.classes, \
                imgsz=self.input_shape, iou=self.detection_iou_thr, conf=self.detection_confidence_thr, \
                max_det=self.max_det, tracker="bytetrack.yaml")
            
            # for detection in detections:
            #     ## Boxes object for bounding box outputs
            #     boxes = detection.boxes
            #     ## Masks object for segmentation masks outputs
            #     # masks = detection.masks
            #     ## Keypoints object for pose outputs
            #     keypoints = detection.keypoints
            #     ## Probs object for classification outputs
            #     probs = detection.probs
            #     ## Oriented boxes object for OBB outputs
            #     obb = detection.obb  
                
            #     logging.info(f'Boxes: {boxes}')
            #     logging.info(f'Keypoints: {keypoints}')
            #     logging.info(f'Classification: {probs}')
            #     logging.info(f'Oriented boxes: {obb}')

            keypoints_list = []

            if self.keypoints_working:
                for i, detection in enumerate(detections[0].boxes.data):

                    if len(detection) != 7:
                        continue
                    
                    x_min, y_min, x_max, y_max = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

                    confidence = float(detection[5])
                    class_id = int(detection[6])

                    if self.class_dict[class_id] == 'person': #class_id == 0:

                        ## Keypoints
                        start_time = datetime.now()
                        cropped_frame = copy.deepcopy(frame)[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if self.save_cropped_person:
                            os.makedirs(self.vis_folder, exist_ok=True)
                            cv2.imwrite(os.path.join(self.vis_folder, f"img_{frame_id}.png"), cropped_frame)
                        keypoints = self.extract_pose(cropped_frame)
                        if keypoints.pose_landmarks:
                            for landmark in keypoints.pose_landmarks.landmark:
                                landmark.x = landmark.x * (x_max - x_min) / frame.shape[1] + x_min / frame.shape[1]
                                landmark.y = landmark.y * (y_max - y_min) / frame.shape[0] + y_min / frame.shape[0]
                        keypoints_list.append(keypoints)
                        end_time = datetime.now()
                        elapsed_time = (end_time - start_time).total_seconds()
                        logging.info(f'Keypoints estimation time per person: {elapsed_time}')     

            return {'detections':detections, 'keypoints': keypoints_list, 'pose_connections':self.mp_pose.POSE_CONNECTIONS}

        elif self.model_version == "ultralytics/ultralytics":
            # detections = self.detector.predict(frame, save=False, classes=self.classes, \
            #     imgsz=self.input_shape, iou=self.detection_iou_thr, conf=self.detection_confidence_thr, \
            #     max_det=self.max_det)
            
            detections = self.detector.track(frame, save=False, classes=self.classes, \
                imgsz=self.input_shape, iou=self.detection_iou_thr, conf=self.detection_confidence_thr, \
                max_det=self.max_det, tracker="bytetrack.yaml", persist=True)
            
            # for detection in detections:
            #     ## Boxes object for bounding box outputs
            #     boxes = detection.boxes
            #     ## Masks object for segmentation masks outputs
            #     # masks = detection.masks
            #     ## Keypoints object for pose outputs
            #     keypoints = detection.keypoints
            #     ## Probs object for classification outputs
            #     probs = detection.probs
            #     ## Oriented boxes object for OBB outputs
            #     obb = detection.obb  
                
            #     logging.info(f'Boxes: {boxes}')
            #     logging.info(f'Keypoints: {keypoints}')
            #     logging.info(f'Classification: {probs}')
            #     logging.info(f'Oriented boxes: {obb}')

            keypoints_list = []
            if self.keypoints_working:
                for i, detection in enumerate(detections[0].boxes.data):
                    if len(detection) != 7:
                        continue
                    
                    x_min, y_min, x_max, y_max = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

                    confidence = float(detection[5])
                    class_id = int(detection[6])

                    if self.detector.names[class_id] == 'person': #class_id == 0:

                        cropped_frame = copy.deepcopy(frame)[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if self.save_cropped_person:
                            os.makedirs(os.path.join(self.vis_folder, video_name), exist_ok=True)
                            cv2.imwrite(os.path.join(self.vis_folder, video_name, f"person_img_{frame_id}.png"), cropped_frame)

                        pose_results = self.pose_estimator.predict(source=cropped_frame, conf=self.pose_confidence_thr)

                        for pose_result in pose_results:
        
                            pose_keypoints = pose_result.keypoints

                            if pose_keypoints.conf is not None:
                                keypoints_xy = pose_keypoints.xy[0].tolist() 
                                keypoints_confidence = pose_keypoints.conf[0].tolist()
                                keypoints_list.append((keypoints_xy, keypoints_confidence, detections[0].boxes[i]))

            return {'detections':detections, 'keypoints': keypoints_list}
