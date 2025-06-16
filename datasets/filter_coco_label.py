import os

def filter_coco_labels(folder_path):
    """
    특정 폴더 안의 COCO 포맷 label text 파일에서 클래스 0~5 인덱스를 제외한 객체 라인을 제거합니다.

    Args:
        folder_path (str): label text 파일들이 있는 폴더 경로.
    """
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            filtered_lines = []
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        parts = line.strip().split()
                        class_index = int(parts[0])  
                        # Set class range
                        if 0 <= class_index <= 5:  
                            filtered_lines.append(line)  
                    except ValueError:
                        print(f"Error processing line in file {filename}: {line.strip()}. 첫 번째 값이 정수형 클래스 인덱스가 아님.")
                    except IndexError:
                        print(f"Error processing line in file {filename}: {line.strip()}. 라인이 비어있거나 형식이 COCO 포맷이 아님.")
            
            with open(file_path, 'w') as file:
                file.writelines(filtered_lines)

            print(f"File {filename} processed: Removed lines with class index outside the range 0-5.")

if __name__ == "__main__":
    
    folder_path = input("label text 파일들이 있는 폴더 경로를 입력하세요: ")
    if not os.path.isdir(folder_path):
        print("Error: 지정된 경로는 폴더가 아닙니다.")
    else:
        filter_coco_labels(folder_path)
        print("COCO label filtering 완료.")
