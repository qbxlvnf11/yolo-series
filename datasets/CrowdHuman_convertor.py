import json
import os
import cv2

# Paths to ODGT files and images
odgt_files = ["datasets/CrowdHuman/annotation_train.odgt", "datasets/CrowdHuman/annotation_val.odgt"]  # Add multiple ODGT files
image_dir = ["datasets/CrowdHuman/CrowdHuman_train/images", "datasets/CrowdHuman/CrowdHuman_val/images"]  # Directory containing images
# output_dir = ["datasets/CrowdHuman_coco_format/labels/train", "datasets/CrowdHuman_coco_format/labels/val"]
output_dir = ["datasets/CrowdHuman_coco_format/labels/train", "datasets/CrowdHuman_coco_format/labels/val"]

class_map = {"person": 0, "head": 80}
#class_map = {"person": 0}

# Create the output directory if it doesn't exist
os.makedirs(output_dir[0], exist_ok=True)
os.makedirs(output_dir[1], exist_ok=True)

# Process each ODGT file
for i, odgt_file in enumerate(odgt_files):
    with open(odgt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)  # Convert JSON string to dictionary
        image_id = data["ID"].split("/")[0]  # Extract the image filename (without extension)
        image_path = os.path.join(image_dir[i], f"{image_id}.jpg")  # Change if image extension differs

        # Read image dimensions
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found! Skipping...")
            continue

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        label_file = os.path.join(output_dir[i], f"{image_id}.txt")

        with open(label_file, "w") as f_out:
            for obj in data["gtboxes"]:
                if obj['tag']=='person':
                    
                    if "head" in class_map:
                        cls_id = class_map["head"]
                        x, y, w, h = obj["hbox"]

                        # Convert to YOLO format (normalized)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # YOLO format: class x_center y_center width height
                        f_out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    
                    cls_id = class_map["person"]
                    x, y, w, h = obj['vbox']

                    # Convert to YOLO format (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height

                    # YOLO format: class x_center y_center width height
                    f_out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete! YOLO labels have been saved.")
