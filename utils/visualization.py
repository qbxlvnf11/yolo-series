import random
import logging
import cv2
import os
import mediapipe as mp
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# matplotlib.use('TkAgg')

# matplotlib.rcParams['font.family'] = 'NanumBarunGothic'
# matplotlib.rcParams['axes.unicode_minus'] = False

from models.utils import get_skeleton_config

def generate_random_colors(num_classes):
    random.seed(42)
    colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(num_classes)
    ]
    return colors

def vis_bounding_box(config, frame, detections, keypoints=None, classes=None, \
        colors=None, mp_drawing=None, pose_connections=None):

    for detection in detections:
        
        track_id = None
        if 'track_id' in detection:
            track_id = detection['track_id']
        
        x_min = int(detection['bbox'][0])
        y_min = int(detection['bbox'][1])
        x_max = int(detection['bbox'][2])
        y_max = int(detection['bbox'][3])

        # confidence = detection['confidence']
        class_id = detection['class_id']
        if event_name is None:
            class_name = classes[class_id]
        else:
            class_name = event_name
        color = colors[class_id]
        text_color = color
        thickness = 1
        
        # label = f"{class_name} {obj_id} ({confidence*100:.1f}%)"
        if track_id is not None:
            label = f"{class_name} (ID={track_id})"
        else:
            label = f"{class_name}"
            
        top_left = (x_min, y_min)
        bottom_right = (x_max, y_max)

        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

    if len(keypoints) > 0: #keypoints.pose_landmarks:
        for keypoint in keypoints:
            mp_drawing.draw_landmarks(
                frame, keypoint.pose_landmarks, pose_connections,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
    
    # detections = sv.Detections.from_inference(results)
    # labels = [classes[class_id] for class_id in detections.class_id]

    # bounding_box_annotator = sv.BoxAnnotator()
    # label_annotator = sv.LabelAnnotator()

    # vis_frame = bounding_box_annotator.annotate(
    #     scene=frame, detections=detections
    # )
    # vis_frame = label_annotator.annotate(
    #     scene=vis_frame, detections=detections, labels=labels
    # )

    return frame

def vis_bounding_box_v2(config, detections, keypoints=None, mp_drawing=None, pose_connections=None, \
    event_name=None, video_name=None, frame_id=-1, vis_folder='vis'):

    frame = detections[0]
    vis_frame = frame.plot(kpt_line=True, font_size=1)
    save_vis = config['save_vis']

    if event_name is not None:
        cv2.putText(vis_frame, event_name, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if len(keypoints) > 0: #keypoints.pose_landmarks:
        for keypoint in keypoints:
            mp_drawing.draw_landmarks(
                vis_frame, keypoint.pose_landmarks, pose_connections,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
    # vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    
    if save_vis:
        # directory = Path(vis_folder)
        # directory.mkdir(exist_ok=True)
        os.makedirs(vis_folder, exist_ok=True)
        # frame.save(filename=os.path.join(vis_folder, f"img_{frame_id}.png"))
        save_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(vis_folder, f"img_{frame_id}.png"), save_frame)

    return vis_frame

def vis_bounding_box_v3(config, detections, keypoints=None, \
    event_flag=False, event_name=None, video_name=None, frame_id=-1, vis_folder='vis'):

    frame = detections[0]
    vis_frame = frame.plot(kpt_line=True, font_size=1)
    # save_vis = config['save_vis']

    if len(keypoints) > 0: #keypoints.pose_landmarks:
        visualize_keypoints_on_original_image(vis_frame, keypoints)
    # vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

    # if event_name is not None:
    #     cv2.putText(vis_frame, event_name, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if event_flag and event_name is not None:
        warning_text = event_name + ' ' + 'detected!!'

        vis_frame = vis_caution(vis_frame, warning_text)

    # if save_vis:
    #     # directory = Path(vis_folder)
    #     # directory.mkdir(exist_ok=True)
    #     os.makedirs(vis_folder, exist_ok=True)
    #     # frame.save(filename=os.path.join(vis_folder, f"img_{frame_id}.png"))
    #     save_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(os.path.join(vis_folder, f"img_{video_name}_{frame_id}.png"), save_frame)

    return vis_frame

def visualize_keypoints_on_original_image(vis_frame, keypoints_info_list):

    skeleton_config, skeleton_color_config, keypoint_names = get_skeleton_config()

    for keypoints_xy, keypoints_confidence, box in keypoints_info_list:
        
        x1 = int(box.data[0][0])
        y1 = int(box.data[0][1])
        x2 = int(box.data[0][2])
        y2 = int(box.data[0][3])
        
        for i, keypoint in enumerate(keypoints_xy):
            kp_x, kp_y = map(int, keypoint)
            if kp_x == 0.0 and kp_y == 0.0:
                continue
            # confidence = keypoints_confidence[i]
            original_kp_x, original_kp_y = kp_x + x1, kp_y + y1
            cv2.circle(vis_frame, (original_kp_x, original_kp_y), radius=5, color=(255, 0, 0), thickness=2)
            # cv2.putText(vis_frame, keypoint_names[i], (original_kp_x + 5, original_kp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for start_joint, end_joint in skeleton_config:
            start_point = keypoints_xy[start_joint]
            end_point = keypoints_xy[end_joint]
            start_x, start_y = map(int, start_point)
            end_x, end_y = map(int, end_point)
            # confidence_start = keypoints_confidence[start_joint]
            # confidence_end = keypoints_confidence[end_joint]

            if (start_x == 0.0 and start_y == 0.0) or (end_x == 0.0 and end_y == 0.0): #confidence_start > 0.5 and confidence_end > 0.5:
                continue

            original_start_x, original_start_y = start_x + x1, start_y + y1
            original_end_x, original_end_y = end_x + x1, end_y + y1

            line_color = (255, 0, 0)

            if (start_joint == 0 and end_joint in [1, 2]) or (start_joint in [1, 2] and end_joint in [3, 4]):
                line_color = skeleton_color_config['face']
            elif (start_joint == 5 and end_joint == 7) or (start_joint == 6 and end_joint == 8) or \
                    (start_joint == 7 and end_joint == 9) or (start_joint == 8 and end_joint == 10):
                line_color = skeleton_color_config['arms']
            elif (start_joint == 11 and end_joint == 13) or (start_joint == 12 and end_joint == 14) or \
                    (start_joint == 13 and end_joint == 15) or (start_joint == 14 and end_joint == 16):
                line_color = skeleton_color_config['legs']
            elif (start_joint == 5 and end_joint == 6) or (start_joint == 11 and end_joint == 12) or \
                    (start_joint == 5 and end_joint == 11) or (start_joint == 6 and end_joint == 12): 
                line_color = skeleton_color_config['torso']
            elif (start_joint == 5 and end_joint == 11) or (start_joint == 6 and end_joint == 12): 
                line_color = skeleton_color_config['shoulder_hip']

            cv2.line(vis_frame, (original_start_x, original_start_y), (original_end_x, original_end_y), color=line_color, thickness=2)

    return vis_frame