import cv2
import os
import threading
import time
import queue
import logging
import yaml
import argparse
from datetime import datetime
from pathlib import Path
import mediapipe as mp
import matplotlib
import matplotlib.pyplot as plt

from utils.utils import get_logger, load_config
from models.model import Model

from utils.visualization import generate_random_colors
from utils.visualization import vis_bounding_box, vis_bounding_box_v2, vis_bounding_box_v3

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration for the application.")

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the configuration file"
    )
    # parser.add_argument(
    #     "--video", 
    #     type=str,
    #     help="Input video or RTSP URL"
    # )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arguments()

    ## Configs, Logs
    config = load_config(args.config)

    now = datetime.now()
    date = now.strftime("%Y%m%d_%H%M%S")

    if 'log_path' in config:
        log_path = config['log_path']
    else:
        directory = Path('logs')
        directory.mkdir(exist_ok=True)
        log_path = os.path.join('logs', "demo_" + date + ".txt")

    logger = get_logger(log_path)

    logger.info('\n'+'- Configs'+'\n'+yaml.dump(config, default_flow_style=False, sort_keys=False))

    ## Setup
    model_version = config['model_version']
    save_video = config['save_video']
    save_vis = config['save_vis']
    save_raw = config['save_raw']

    mp_drawing = mp.solutions.drawing_utils
    video_writer = None

    img_vis_folder = 'vis'
    
    ax = plt.subplot(1,1,1)
    set_flag = False

    ## Models
    classes = config['detection']['classes']
    colors = generate_random_colors(len(classes))

    model = Model(config, classes, vis_folder=f'vis/vis_{date}/persons')

    video_list = config['video_list']
    for video_id, video in enumerate(video_list):
        
        logger.info(f'Video ID: {video_id}')
        logger.info(f'Video Name: {video}')
        base_path, old_extension = os.path.splitext(video)
        video_name = os.path.basename(base_path)

        frame_id = 1

        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            logger.error("Error: Unable to open RTSP, VIDEO stream")
            continue

        # FPS
        ori_fps = cap.get(cv2.CAP_PROP_FPS)
        if 'fps' in config:
            fps = min(config['fps'], ori_fps) #cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(ori_fps / fps)
        else:
            fps = ori_fps
            frame_interval = 1

        #try:
        while True:
            vis_frame = None
            ret, frame = cap.read()

            if frame_id == 1:
                logger.info("Start to read frame!")

            if save_raw:
                # directory = Path(vis_folder)
                # directory.mkdir(exist_ok=True)
                os.makedirs(os.path.join(img_vis_folder, video_name), exist_ok=True)
                # frame.save(filename=os.path.join(vis_folder, f"img_{frame_id}.png"))
                #save_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(img_vis_folder, video_name, f"raw_img_{video_name}_{frame_id}.png"), frame)

            if not ret:        
                logger.error("Error: Unable to read frame from RTSP stream")
                break
            
            ori_height, ori_width, _ = frame.shape
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if 'input_shape' in config:
                dim = (config['input_shape']['width'], config['input_shape']['height'])
                frame = cv2.resize(frame, dim)

            if frame_id % frame_interval == 0:
                results = model.inference(frame, frame_id)

                if model_version == 'yolo_world/deep_sort/mp_pose':
                    keypoints = results['keypoints']
                    pose_connections = results['pose_connections']
                    
                    vis_frame = vis_bounding_box(config, frame, detections, keypoints, \
                        classes, colors, mp_drawing, pose_connections)
                    # vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

                elif model_version == "ultralytics/mp_pose":
                    detections = results['detections']
                    keypoints = results['keypoints']
                    pose_connections = results['pose_connections']
                    
                    vis_frame = vis_bounding_box_v2(config, detections, keypoints, \
                        mp_drawing, pose_connections, \
                        frame_id=frame_id, vis_folder=img_vis_folder)

                elif model_version == "ultralytics/ultralytics":
                    detections = results['detections']
                    keypoints = results['keypoints']
                    
                    vis_frame = vis_bounding_box_v3(config, detections, keypoints, \
                        video_name=video_name, frame_id=frame_id) #, vis_folder=img_vis_folder)

                #logger.info(f"Detections: {detections[0].boxes.data}")
                vis_frame = cv2.resize(vis_frame, (ori_width, ori_height))

                if save_vis:
                    # directory = Path(vis_folder)
                    # directory.mkdir(exist_ok=True)
                    os.makedirs(os.path.join(img_vis_folder, video_name), exist_ok=True)
                    # frame.save(filename=os.path.join(vis_folder, f"img_{frame_id}.png"))
                    #save_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(img_vis_folder, video_name, f"img_{video_name}_{frame_id}.png"), vis_frame)

                if save_video:

                    # directory = Path(video_vis_folder)
                    # directory.mkdir(exist_ok=True)
                    
                    if video_writer is None:
                        os.makedirs(video_vis_folder, exist_ok=True)
                        output_video_path = os.path.join(video_vis_folder, f"video_{video_name}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (ori_width, ori_height))
                        if not video_writer.isOpened():
                            logger.error(f"Error: VideoWriter: {output_video_path}")
                    
                    #save_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    video_writer.write(vis_frame)

            if frame_id % 100 == 0:
                logger.info(f"Frame ID: {frame_id}")

            frame_id += 1

            ## Open board
            if vis_frame is not None and not set_flag:
                im = ax.imshow(vis_frame)
                set_flag = True

            ## Change data
            if vis_frame is not None:
                im.set_data(vis_frame)
                
            # Wait
            plt.pause(0.0000001)

        '''
        except Exception as e:
            cap.release()
            logger.error(f"Exception in {video_id}: {video} - {e}") 
        finally:
            cap.release()
            logger.info(f"Finished processing video {video_id}: {video}") 
        '''
