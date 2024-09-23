import torch
import cv2

if __name__ == "__main__":
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().cuda()
    image = cv2.imread("image.jpg")
    image_tensor = dinov2.preprocess(image)
    with torch.no_grad():
        features = dinov2.forward(image_tensor)
    print(features)
