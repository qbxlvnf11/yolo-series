import json
import os

def labelme_to_coco_txt(labelme_json_path, coco_txt_path):
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    annotations_info = []

    image_height = labelme_data['imageHeight']
    image_width = labelme_data['imageWidth']

    for shape in labelme_data['shapes']:
        if shape['label'] == 'person' and shape['shape_type'] == 'rectangle':
            points = shape['points']
            x_min = int(min(points[0][0], points[1][0]))
            y_min = int(min(points[0][1], points[1][1]))
            x_max = int(max(points[0][0], points[1][0]))
            y_max = int(max(points[0][1], points[1][1]))

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            if bbox_width > 0 and bbox_height > 0:
                # Normalization 적용 (0~1 범위)
                x_center_norm = ((x_min + x_max) / 2) / image_width
                y_center_norm = ((y_min + y_max) / 2) / image_height
                width_norm = bbox_width / image_width
                height_norm = bbox_height / image_height

                annotation_line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                annotations_info.append(annotation_line)

    with open(coco_txt_path, 'w') as outfile:
        outfile.writelines(annotations_info)

def convert_folder_labelme_to_coco_txt(labelme_folder_path, coco_folder_path):
    if not os.path.exists(coco_folder_path):
        os.makedirs(coco_folder_path)

    json_files = [f for f in os.listdir(labelme_folder_path) if f.endswith('.json')]

    if not json_files:
        return

    for json_file in json_files:
        labelme_json_path = os.path.join(labelme_folder_path, json_file)
        coco_txt_name = os.path.splitext(json_file)[0] + ".txt"
        coco_txt_path = os.path.join(coco_folder_path, coco_txt_name)

        labelme_to_coco_txt(labelme_json_path, coco_txt_path)


if __name__ == "__main__":
    labelme_folder_path = "datasets/custom_labeling"
    coco_folder_path = "datasets/custom_labeling_coco_format/labels/train"

    if not os.path.exists(labelme_folder_path):
        os.makedirs(labelme_folder_path)

    convert_folder_labelme_to_coco_txt(labelme_folder_path, coco_folder_path)

    coco_output_example_path = os.path.join(coco_folder_path, os.listdir(coco_folder_path)[0]) if os.listdir(coco_folder_path) else None
    if coco_output_example_path and os.path.exists(coco_output_example_path):
        print(f"\nCOCO format TXT 파일 (coco_txts_normalized 폴더) 내용 (첫 번째 파일 예시 - {coco_output_example_path}):") # 폴더 이름 변경
        with open(coco_output_example_path, 'r') as f:
            coco_output_data = f.readlines()
            for line in coco_output_data:
                print(line.strip())
    else:
        print("COCO TXT 파일이 생성되지 않았거나, 폴더가 비어 있습니다.")