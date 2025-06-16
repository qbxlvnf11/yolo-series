import torch

def compare_models(model1, model2):
    layers_model1 = list(model1.named_children())
    layers_model2 = list(model2.named_children())
    
    same_layers = []
    diff_layers = []
    
    for (name1, layer1), (name2, layer2) in zip(layers_model1, layers_model2):
        if type(layer1) == type(layer2):
            same_layers.append((name1, layer1))
        else:
            diff_layers.append((name1, (layer1, layer2)))
    
    return same_layers, diff_layers

def get_skeleton_config():

    skeleton_config = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (5, 11), (6, 12)
    ]

    skeleton_color_config = {
        'face': (0, 255, 255),     
        'arms': (0, 255, 0),     
        'legs': (255, 0, 0),     
        'torso': (255, 0, 255),   
        'shoulder_hip': (255, 255, 0) 
    }

    keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    return skeleton_config, skeleton_color_config, keypoint_names
