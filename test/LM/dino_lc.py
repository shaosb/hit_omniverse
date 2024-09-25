import torch
import cv2
from torchvision import transforms

if __name__ == "__main__":
    image = cv2.imread("image.jpg")
    patch_size = 14

    H, W, _ = image.shape
    patch_h = int(H // patch_size)
    patch_w = int(W // patch_size)
    new_H = patch_h * patch_size
    new_W = patch_w * patch_size

    image = cv2.resize(image, (new_W, new_H))

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classifier = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").eval().cuda()
    
    image = transform(image).unsqueeze(0).cuda()
    result = classifier(image)#.argmax(dim=1).squeeze(0).item()
    classification = result.argmax(dim=1).squeeze(0).item()
    print(classification)
