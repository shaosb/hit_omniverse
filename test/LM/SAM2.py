import torch
import cv2
from sam2.sam2_image_predictor import SAM2ImagePredictor


if __name__ == "__main__":
    image = cv2.imread("image.jpg")
    
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict()
