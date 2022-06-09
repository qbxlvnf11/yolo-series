from __future__ import division
import time
import tqdm
import numpy as np
import cv2
import collections
import os

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torchvision
from torchvision.ops import nms as nms
import torchvision.transforms as transforms
import torch.nn.functional as F

# ---------- Filtering classes ---------

def load_classes(path):
	"""
	Loads class labels at 'path'
	"""
	fp = open(path, "r")
	names = fp.read().split("\n")[:-1]
	return names

def filter_classes(detections, classes):
	mask = torch.stack([torch.stack([detections[:, -1] == cls]) for cls in classes])
	mask = torch.sum(torch.squeeze(mask, dim=1), dim=0)
	return detections[mask > 0]

# ---------- Filtering classes ---------

# ---------- Preprocessing images & labels (object boxes) for detection ---------

def xywh2xyxy_np(x): # (center x, center y, width, height) to (left top x, left top y, right bottom x, right bottom y)
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
    return image

class ImgAug(object):
    def __init__(self, augmentations=[], return_xywh=True):
        self.augmentations = augmentations
        self.return_xywh = return_xywh

    def __call__(self, data):
        # Unpack data
        img, boxes = data
        
        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])
        
        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)
        
        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)
        
        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            
            # (x, y, x, y)
            if self.return_xywh:
                boxes[box_idx, 0] = box.label
                boxes[box_idx, 1] = ((x1 + x2) / 2)
                boxes[box_idx, 2] = ((y1 + y2) / 2)
                boxes[box_idx, 3] = (x2 - x1)
                boxes[box_idx, 4] = (y2 - y1)

            # (center x, center y, width, height)
            else:    
                boxes[box_idx, 0] = box.label
                boxes[box_idx, 1] = x1
                boxes[box_idx, 2] = y1
                boxes[box_idx, 3] = x2
                boxes[box_idx, 4] = y2
                        
        return img, boxes


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class PadSquare(ImgAug):
    def __init__(self, return_xywh=True):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])
        self.return_xywh = return_xywh


class DefaultAug(ImgAug):
    def __init__(self, return_xywh=True):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])
        self.return_xywh = return_xywh


class StrongAug(ImgAug):
    def __init__(self, return_xywh=True):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])
        self.return_xywh = return_xywh


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


# Processing only image in inference
inf_image_transform = transforms.Compose([
		PadSquare(),
		ToTensor()])

# Processing image & absolute label in valid	
valid_image_absolute_label_transform = transforms.Compose([
		PadSquare(return_xywh=False),
		RelativeLabels(),
		ToTensor()])

# Processing image & relative label in valid	
valid_image_relative_label_transform = transforms.Compose([
		AbsoluteLabels(),
		PadSquare(return_xywh=False),
		RelativeLabels(),
		ToTensor()])

# Processing image & absolute label in train		
train_image_absolute_label_transform = transforms.Compose([
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

# Processing image & relative label in train		
train_image_relative_label_transform = transforms.Compose([
		AbsoluteLabels(),
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

# Processing image & absolute label with default augmentation in train
train_default_augmentation_image_absolute_label_transform = transforms.Compose([
		DefaultAug(),
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

# Processing image & absolute label with strong augmentation in train
train_strong_augmentation_image_absolute_label_transform = transforms.Compose([
		StrongAug(),
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

# Processing image & relative label with default augmentation in train
train_default_augmentation_image_relative_label_transform = transforms.Compose([
		AbsoluteLabels(),
		DefaultAug(),
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

# Processing image & relative label with strong augmentation in train
train_strong_augmentation_image_relative_label_transform = transforms.Compose([
		AbsoluteLabels(),
		StrongAug(),
		PadSquare(),
		RelativeLabels(),
		ToTensor()])

def is_gray_scale_image(img):

	if len(img.shape) == 3 and np.std([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])]) < 0.01:
		return True
	elif len(img.shape) == 2:
		return True
	    
	return False
		
def processing_detection_image(image, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor = inf_image_transform((image, np.zeros((1, 5))))[0]
	images_tensor = resize(images_tensor, resize_width_height)
		
	return images_tensor

def processing_valid_detection_image_absolute_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = valid_image_absolute_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor
	
def processing_valid_detection_image_relative_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = valid_image_relative_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

def processing_detection_image_absolute_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_image_absolute_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

def processing_detection_image_default_augmentation_absolute_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_default_augmentation_image_absolute_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor
	
def processing_detection_image_strong_augmentation_absolute_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_strong_augmentation_image_relative_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

def processing_detection_image_relative_labels(image, object_boxes_np, resize_width_height):

	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_image_relative_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

def processing_detection_image_default_augmentation_relative_labels(image, object_boxes_np, resize_width_height):

	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_default_augmentation_image_relative_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

def processing_detection_image_strong_augmentation_relative_labels(image, object_boxes_np, resize_width_height):
	
	gray_scale = is_gray_scale_image(image)
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	images_tensor, object_boxes_tensor = train_strong_augmentation_image_relative_label_transform((image, object_boxes_np))
	images_tensor = torch.squeeze(resize(images_tensor, resize_width_height), dim=0)
	
	return images_tensor, object_boxes_tensor

# ---------- Preprocessing images & labels (object boxes) for detection ---------

# ---------- NMS ---------

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
    
def non_max_suppression(prediction, conf_thres, iou_thres, xywh=True):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    
    nc = prediction.shape[2] - 5  # number of classes (e.g. 80 = one hot vector of object classes)

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        if xywh: # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
        else: # (x1, y1, x2, y2)
            box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def scale_coords(coords, from_image_shape, to_image_shape):
	# Rescale coords (xyxy) from from_image_shape to to_image_shape
	gain = max(from_image_shape) / max(to_image_shape)  # gain  = old / new
	coords[:, [0, 2]] -= (from_image_shape[1] - to_image_shape[1] * gain) / 2  # x padding
	coords[:, [1, 3]] -= (from_image_shape[0] - to_image_shape[0] * gain) / 2  # y padding
	coords[:, :4] /= gain
	coords[:, :4] = coords[:, :4].clamp(min=0)
	return coords

# ---------- NMS ---------

# --------- visualization ---------

def draw_boxes(img, box_list, class_list, inf_img_path, save_path):
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	color = (0, 0, 255) # red
	thickness = 2 # thickness of line
	
	for box, class_name in zip(box_list, class_list):
		# xyxy format
		x1, y1, x2, y2 = list(map(int, box))

		t_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
		cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
		cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
		cv2.putText(img, class_name, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
	
	name = inf_img_path.split('/')[-1].split('.')[0]+'_inferencing_yolov3.png'

	if not os.path.exists(save_path):
		print('Make folder {}'.format(save_path))
		os.makedirs(save_path)
	cv2.imwrite(os.path.join(save_path, name), img)
	
	print(img.shape)
	print('Complete save inferencing image!', os.path.join(save_path, name))
		
# --------- visualization ---------
