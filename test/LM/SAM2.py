import torch
import cv2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import time

if __name__ == "__main__":
    image = cv2.imread("house.jpg")
    image = cv2.resize(image, (640, 480))
    point = [300, 300]

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        predictor.set_image(image)

        start = time.time()
        masks, _, _ = predictor.predict(
            # point_coords=np.array([[673, 512]]),
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        print(f"spend tims:{time.time() - start}")

        # 为每个mask生成随机颜色
        colors = [np.random.randint(0, 255, 3).tolist() for _ in range(len(masks))]
        
        # 创建一个与原图相同大小的透明图层
        overlay = np.zeros(image.shape, dtype=np.uint8)
        
        # 遍历所有mask
        for i, mask in enumerate(masks):
            # 将mask转换为uint8类型
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            point_color = (0, 0, 255)  # 红色
            point_size = 10
            point_thickness = -1  # 填充圆
            cv2.circle(image, tuple(point), point_size, point_color, point_thickness)
            
            # 在记号周围绘制一个白色边框，使其更加醒目
            border_color = (255, 255, 255)  # 白色
            border_thickness = 2
            cv2.circle(image, tuple(point), point_size + border_thickness, border_color, border_thickness)


            # 为每个mask生成不同的透明度
            alpha = np.random.uniform(0.3, 0.7)
            
            # 在overlay上绘制填充的mask
            overlay_color = overlay.copy()
            overlay_color[mask_uint8 > 0] = colors[i]
            
            # 将填充的mask与原图混合
            cv2.addWeighted(overlay_color, alpha, image, 1 - alpha, 0, image)
            
            # 在mask边缘绘制轮廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, colors[i], 2)
        
        # 显示结果
        cv2.imshow("带有Masks的图像", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        cv2.imwrite("image_with_masks.jpg", image)
        print("带有Masks的图像已保存为'image_with_masks.jpg'")