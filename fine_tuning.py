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

from utils.utils import get_logger, load_config
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration for the application.")

    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the configuration file"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)

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
        log_path = os.path.join('logs', 'train_' + date + ".txt")
    args = parse_arguments()

    logger = get_logger(log_path)
    logger.info('\n'+'- Configs'+'\n'+yaml.dump(config, default_flow_style=False, sort_keys=False))
    
    ## Models
    model = YOLO(config['detection']['ultralytics_model_weights'])
    model.info()

    ## Train
    epochs = config['epochs']
    batch_size = config['batch_size']
    imgsz = config['imgsz']
    device = config['device']

    results = model.train(data=args.config, epochs=epochs, \
        batch=batch_size, imgsz=imgsz, device=device, \
        task="detect")

