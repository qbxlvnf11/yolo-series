import logging

def xywh_to_xyxy(x, y, width, height):

    x_min = int(x - width / 2)
    y_min = int(y - height / 2)
    x_max = int(x_min + width)
    y_max = int(y_min + height)
        
    return x_min, y_min, x_max, y_max

def convert_bbox_coordinates(bbox, ori_width, ori_height, target_width=640, target_height=640):
    
  x_min, y_min, x_max, y_max = bbox

  # x 좌표 변환
  x_min = x_min * ori_width / target_width
  x_max = x_max * ori_width / target_width

  # y 좌표 변환
  y_min = y_min * ori_height / target_height
  y_max = y_max * ori_height / target_height

  return [x_min, y_min, x_max, y_max]
