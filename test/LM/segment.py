from SAM2 import LVM
from gpt import LargeModel
import cv2
import re

def middleware(command):
    pattern = r'(\w+):\[?\((\d+),(\d+)\)(?:,\s*\((\d+),(\d+)\))?\]?'
    matches = re.findall(pattern, command)

    result_dict = {}
    for match in matches:
        key = match[0]
        if match[3] and match[4]:
            value = [(int(match[1]), int(match[2])), (int(match[3]), int(match[4]))]
        else:
            value = (int(match[1]), int(match[2]))
        result_dict[key] = value

    points = []
    for key, value in result_dict.items():
        print(f"{key}: {value}")
        if isinstance(value, list):
            for v in value:
                points.append(v)
            continue
        points.append(value)
    
    return points

if __name__ == "__main__":
    image_path = "./val/109.jpg"
    gpt = LargeModel("prompt_segment.txt")
    sam2 = LVM()

    command = gpt.talk_to_gpt("search for people", image_path)
    points = middleware(command)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))

    masks = sam2.predict(image, points)
    sam2.show_mask(image, masks, points, save_path="output.jpg", show=True)