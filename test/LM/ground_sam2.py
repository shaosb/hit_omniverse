import cv2
import torch
import supervision as sv
import pycocotools.mask as mask_util
import numpy as np
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import yaml

class GroundingDINO_SAM:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.GROUNDING_MODEL = config["Dino"]["grounding_model"]
        self.SAM2_CHECKPOINT = config["SAM2"]["checkpoint"]
        self.SAM2_MODEL_CONFIG = config["SAM2"]["model_config"]
        self.DINO_BOX_THRESHOLD = config["Dino"]["box_threshold"]
        self.DINO_TEXT_THRESHOLD = config["Dino"]["text_threshold"]
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        torch.autocast(device_type=self.DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.SAM2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT, device=self.DEVICE)
        self.SAM2_predictor = SAM2ImagePredictor(self.SAM2_model)

        self.processor = AutoProcessor.from_pretrained(self.GROUNDING_MODEL)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.GROUNDING_MODEL).to(self.DEVICE)

    
    def get_grounding_dino_boxes(self, image, text):
        image: Image

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.DEVICE)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.DINO_BOX_THRESHOLD,
            text_threshold=self.DINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )

        return results
    
    def get_sam2_masks(self, image, boxes):
        image: Image

        self.SAM2_predictor.set_image(np.array(image.convert("RGB")))

        masks, scores, logits = self.SAM2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits

    def get_finesegment_feature_field(self, image, text, visualize=True, save_path="default"):
        image: Image

        try:
            prediction = self.get_grounding_dino_boxes(image, text)
            boxes = prediction[0]["boxes"].cpu().numpy()
            masks, _, _ = self.get_sam2_masks(image, boxes)

            if visualize is True or save_path is not None:
                self.save_visualization(image, masks, prediction, visualize, save_path)

        except Exception as e:
            print(f"Error: {e}")
            return None

    def save_visualization(self, image, masks, prediction, visualize, save_path):
        image: Image

        image = np.array(image)
        input_boxes = prediction[0]["boxes"].cpu().numpy()
        confidences = prediction[0]["scores"].cpu().numpy().tolist()
        class_names = prediction[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        masked_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        if save_path is not None:
            cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)
            cv2.imwrite("grounded_sam2_annotated_image_with_mask.jpg", masked_frame)

        if visualize:
            cv2.imshow("GroundingDINO_SAM", masked_frame)
            cv2.imshow("GroundingDINO", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

if __name__ == "__main__":
    image_path = "gpt_input_image.jpg"
    text = "floor. people. obstacle."
    image = Image.fromarray(cv2.imread(image_path))
    Splitter = GroundingDINO_SAM()
    Splitter.get_finesegment_feature_field(image, text, visualize=True)
