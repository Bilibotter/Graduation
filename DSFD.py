import cv2
import sys
import torch
import numpy as np
from torch import device
sys.path.append('./DSFDApply')
from face_ssd import SSD
from base import Detector
from build import DETECTOR_REGISTRY
from config import resnet152_model_config


@DETECTOR_REGISTRY.register_module
class DSFD_Detector(Detector):
    def __init__(self, confidence_threshold=.2, nms_iou_threshold=0.3, use_device=device(type='cuda')):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = use_device
        self.max_resolution = None
        self.mean = np.array(
            [123, 117, 104], dtype=np.float32).reshape(1, 1, 1, 3)
        state_dict = torch.load(r"./weights/WIDERFace_DSFD_RES152.pth")
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,):
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        boxes = self.net(
            x, self.confidence_threshold, self.nms_iou_threshold
        )
        return boxes

    def get_boxes(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = image[None]
        boxes = self.batched_detect(image)[0]
        return [list(map(int, box)) for box in boxes]


if __name__ == '__main__':
    path = r'E:\image\png\21909978_2045803535651443_8493846200674418688_n.jpg'
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    shape = image.shape
    multiply = max(image.shape) / 400
    img = cv2.resize(image, (int(shape[1] / multiply), int(shape[0] / multiply)))
    d = DSFD_Detector()
    boxes = d.get_boxes(image)[0]
    print(boxes)
