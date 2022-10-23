import torch
import torch.nn as nn

import numpy as np
import math
import contextlib
import io
import itertools
import json
import tempfile
import time
import copy
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate

from yolox.utils import postprocess

def eval(device, model, loader, resize_width_height, class_list, class_num, nms_thres, object_confidence_threshold, val_iou_threshold):
        
        model.eval()
		
        target_list = []
        sample_metrics = []
        total_detected_boxes_len = 0
		
        for batch_idx, (img, target, img_info) in enumerate(loader):	

            with torch.no_grad():	
                
                # Image
                img = torch.from_numpy(img).to(device) # e.g. (4, 3, 640, 640)
                
                # Lable: number of labels / img_id, x1, y1, x2, y2, class       		
                target = torch.from_numpy(target).to(device) # e.g. (81, 6)
                
                # Outputs: 1 / number of proposals / (center_x, center_y, w, h, conf) + (boxes, conf) + number of class names
                output = model(img) # e.g. (4, 8400, 85)
                	
                # Postprocessing: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                detections = postprocess(output, class_num, object_confidence_threshold, nms_thres) # e.g. (1, 4, 7)
                #detections = [detection.cpu().numpy() if detection is not None else detection for detection in detections]
				
                # Class list
                target_list += copy.deepcopy(target).cpu().numpy()[:, 5].tolist()
			
            # Batch statistics
            batch_metrics, detected_boxes_len = get_batch_statistics(detections, target, iou_threshold=val_iou_threshold)
            
            sample_metrics += batch_metrics
            total_detected_boxes_len += detected_boxes_len
							
            if len(sample_metrics) == 0:  # No detections over whole validation set.
                print("---- No detections over whole validation set ----")
                return None
		
            target_list = np.array(target_list)
            class_unique_list = np.unique(target_list)
            print('class_unique_list:', class_unique_list)
		
            # Concatenate sample statistics
            true_positives, pred_scores, pred_classes = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
			
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_classes, target_list)
		
            # mAP
            mAP = print_eval_stats(metrics_output, class_list)
            
            return mAP
                
def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    detected_boxes = []
    # Number of wrong predicted class
    
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 5].cpu().numpy()
        pred_labels = output[:, -1].cpu().numpy()
	
        true_positives = np.zeros(pred_boxes.shape[0])
        
        #targets = targets.cpu()
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 4] if len(annotations) else []
        
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, :4]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
                
                # Ignore if label is not one of the target labels
                if pred_label not in target_labels.cpu().numpy():
                    continue
                
                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))
                              
                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou_eval(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]
                
                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    
    return batch_metrics, len(detected_boxes)

def bbox_iou_eval(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
    
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
	
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    print('F1 score:', f1)

    return p, r, ap, f1, unique_classes.astype("int32")
 
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

    
def print_eval_stats(metrics_output, class_names, verbose=True):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print('ap_table:', ap_table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

    return AP.mean()
