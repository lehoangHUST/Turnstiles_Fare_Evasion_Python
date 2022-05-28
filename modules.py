# general import
import torch
import torch.backends.cudnn as cudnn

import os, sys

# YOLOv5
# ------------------------------------------------------------
sys.path.insert(0, "detections/yolov5")

from models.common import DetectMultiBackend
from utils.general import check_img_size

sys.path.insert(0, "utils/efficient_pose")


def config_Yolov5(yolo_weight, device, imgsz=640):
    # Load model
    model = DetectMultiBackend(yolo_weight, device=device)  # load FP32 model
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    return model, stride, names, imgsz