import os
import xml.etree.ElementTree as ET

# Define the input directory containing XML files
input_dir = "datasets/Safety_Helmet_Detection_with_Extended_Labels/Annotations"

# Define the output directory for YOLO label files
# output_dir = "datasets/Safety_Helmet_Detection_with_Extended_Labels/Annotations_convert"
output_dir = "datasets/Safety_Helmet_Detection_with_Extended_Labels_coco_format/labels"

# Mapping of class names to YOLO class indices
class_map = {"person_no_helmet": 0, "person_with_helmet": 0, "head": 80, "head_with_helmet": 81, "helmet": 82}

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process each XML file in the input directory
for file in os.listdir(input_dir):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(input_dir, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image size information
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    label_data = []

    # Iterate over all objects in the XML file
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Process only specific classes
        if class_name not in class_map:
            continue

        class_id = class_map[class_name]

        # Extract bounding box coordinates
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / (2.0 * img_width)
        y_center = (ymin + ymax) / (2.0 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Store the label in YOLO format
        label_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save the label file (same filename, .txt extension)
    if label_data:
        output_file = os.path.join(output_dir, file.replace(".xml", ".txt"))
        with open(output_file, "w") as f:
            f.write("\n".join(label_data))

print("Conversion completed successfully!")
