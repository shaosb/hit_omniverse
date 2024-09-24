import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate
from sklearn.decomposition import PCA

class LVM:
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().cuda()
        self.patch_size = 14
        
    def _preprocess(self, image):
        H, W, _ = image.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_image = cv2.resize(image, (new_W, new_H))
        transformed_image = transformed_image.astype(np.float32) / 255.0
        shape_info = {"img_h": H, 
                      "img_w": W, 
                      "patch_h": patch_h, 
                      "patch_w": patch_w}
        return transformed_image, image, shape_info

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def forward(self, image):
        transformed_image, image, shape_info = self._preprocess(image)
        img_tensors = torch.from_numpy(transformed_image).permute(2, 0, 1).unsqueeze(0).to("cuda:0")  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"

        features = self.model.forward_features(img_tensors)
        raw_feature_grid = features['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, shape_info["patch_h"], shape_info["patch_w"], -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]

        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(shape_info["img_h"], shape_info["img_w"]),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1]) 

        return features_flat, shape_info
    
    def visualize_features(self, features, shape_info):
        features_np = features.cpu().numpy()
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_np)
        features_normalized = ((features_pca - features_pca.min()) / (features_pca.max() - features_pca.min()) * 255).astype(np.uint8)
        feature_image = features_normalized.reshape(shape_info["img_h"], shape_info["img_w"], 3)
        return feature_image

if __name__ == "__main__":
    LVM = LVM()
    image = cv2.imread("image.jpg")
    features, shape_info = LVM.forward(image)
    feature_image = LVM.visualize_features(features, shape_info)

    cv2.imshow("feature_visualization", feature_image)
    cv2.waitKey(0)
    cv2.imwrite("feature_visualization.png", feature_image)
    print("Save image at 'feature_visualization.png'")