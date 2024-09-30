from gpt import LargeModel
from ground_sam2 import GroundingDINO_SAM
import cv2


if __name__ == "__main__":
    GPT = LargeModel("prompt_segment.txt")
    LVM = GroundingDINO_SAM()

    image_path = "gpt_input_image.jpg"

    command = GPT.talk_to_gpt("Search for people", image_path)
    print(command)
    LVM.get_finesegment_feature_field(image_path, command, visualize=True)
    
