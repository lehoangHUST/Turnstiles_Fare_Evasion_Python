from __future__ import division, print_function, absolute_import
import ctypes
import os
import shutil
import cv2
import numpy as np
import modules
import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn

# Code detect human with file weights Yolov5
from detections.yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS
from detections.yolov5.utils.torch_utils import time_sync, select_device
from detections.yolov5.utils.general import set_logging, non_max_suppression, xyxy2xywh, scale_coords
from detections.yolov5.utils.plots import Annotator

from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker

from pose_estimations.efficient_pose.pytorch_model import EfficientPose

import generate_detections as gdet
from vf_logic_checking import tlbr2tlwh, cropImage, VirtualFence

POINTS = ((350, 1400), (950, 1400))  # A and B for cam 2
FORWARD_FLOW = 1  # =0 if (-) -> (+); =1 if (+) -> (-)
VF_OFFSET = 20  # pixels
virtual_fence = VirtualFence(POINTS[0], POINTS[1], VF_OFFSET)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--savevid', action="store_true")
    parser.add_argument('--savename', type=str, default="/content/video.avi")
    parser.add_argument('--human_weight', type=str, default="/content/gdrive/MyDrive/model/v5s_human_mosaic.pt")
    args = parser.parse_args()
    return args


@torch.no_grad()
def run(args):
    human_weight, device = args.human_weight, args.device

    print(torch.cuda.is_available())
    # Initialize
    set_logging()
    device = select_device(device)

    # Load nets YOLOv5
    net_YOLO, strides, yolo_name, imgsz = modules.config_Yolov5(human_weight, device)

    # Load efficientPose
    efficient_pose = EfficientPose(
        folder_path=r"/content/bin_imgs", model_name="I")

    # Load tracker / deepsort
    NN_BUDGET = None
    MAX_COSINE_DISTANCE = 0.3
    TRACKING_MODEL = r"/content/Turnstiles_Fare_Evasion_Python/models/deep_sort/mars-small128.pb"
    encoder = gdet.create_box_encoder(TRACKING_MODEL, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric)

    # Load data
    # Re-use yolov5 data loading pipeline for simplicity
    webcam = args.source.isnumeric()
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=imgsz,
                              stride=strides)  # (sources, letterbox_img: np, orig_img: cv2, None)
    else:
        dataset = LoadImages(args.source, img_size=imgsz,
                             stride=strides)  # (path, letterbox_img: np, orig_img: cv2, cap)

        # saving prediction video

    if args.savevid:
        width = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_WIDTH)
        height = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_HEIGHT)
        res = (int(width), int(height))
        # this format fail to play in Chrome/Win10/Colab
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # codec
        # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
        output = cv2.VideoWriter(args.savename, fourcc, 30, res)

    # Run Inference
    for index, (path, im, im0s, vid_cap, _) in enumerate(dataset):
        is_img = True if any(ext in path for ext in IMG_FORMATS) else False
        annotator = Annotator(np.ascontiguousarray(im0s),
                              line_width=2,
                              font_size=1)

        # yolo inference
        # -----------------------------------------
        t1 = time_sync()
        im_yolo = torch.from_numpy(im).to(device)  # yolo input
        im_yolo = im_yolo.float()
        im_yolo /= 255
        if len(im_yolo.shape) == 3:
            im_yolo = im_yolo[None]  # expand for batch dim
        t2 = time_sync()
        # time logging for data loading
        dt0 = t2 - t1
        # Inference on yolov5
        yolo_preds = net_YOLO(im_yolo)  # (batch, (bbox, conf, class)) type torch.Tensor
        t3 = time_sync()
        # print(f"YOLO inference time: {t3 - t2:.4f}")
        # time logging for yolo predicting
        # nms for yolo
        # yolo_preds: torch.Tensor
        yolo_preds = non_max_suppression(yolo_preds, 0.6, 0.5, [0], None, max_det=100)[0]
        t4 = time_sync()
        # nms time for yolo
        # print(f"YOLO nms time: {t4 - t3:.4f}")

        # scale yolo preds to im0 for drawing
        if len(yolo_preds):
            yolo_preds[:, :4] = scale_coords(im_yolo.shape[2:], yolo_preds[:, :4], im0s.shape).round()

        """
        bbox_preds, conf_preds, cls_id_pred = yolo_preds[:, :4].type(torch.IntTensor), yolo_preds[:, 4], yolo_preds[:, 5]

        features = encoder(im0s, boxes=bbox_preds)
        detections = [Detection(box, confidence, feature)
                      for box, confidence, feature
                      in zip(bbox_preds, conf_preds, features)]

        # Update tracker
        tracker.predict()
        tracker.update(detections)

        # Update current state of each track in tracks
        virtual_fence.updateCurrentStates(tracker.tracks)
        virtual_fence.drawLine(im0s, (225, 235, 25))

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # draw confirmed tracks
            track_box = track.to_tlbr()
            cv2.rectangle(im0s, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])),
                         [0, 153, 255], 2)
            cv2.putText(im0s, "ID: " + str(track.track_id), (int(track_box[0]), int(track_box[1])), 0, 1,
                       (0, 0, 255), 1)

            if track.pre_state == None:
                track.pre_state = track.current_state
                continue
            if track.pre_state == track.current_state:
                continue

            # Xem label của object khi đã qua VirtualFence
            if ((not FORWARD_FLOW and track.pre_state == 1 and track.current_state == -1) \
                    or (FORWARD_FLOW and track.pre_state == -1 and track.current_state == 1)):
                cv2.rectangle(im0s, (int(track_box[0]), int(track_box[1])),
                             (int(track_box[2]), int(track_box[3])), [0, 255, 255], 4)
                cv2.putText(im0s, "Warning: backward flow", (int(track_box[0]), int(track_box[1]) + 24), 0, 1,
                           [0, 255, 255], thickness=2, lineType=cv.LINE_AA, )
                track.pre_state = track.current_state
                continue
            else:
                offset = 20
                h_start = max(0, int(track_box[1]) - offset)
                h_end = min(int(track_box[3]) + offset, frame.shape[0])
                w_start = max(0, int(track_box[0]) - offset)
                w_end = min(int(track_box[2]) + offset, frame.shape[1])
                temp_img = frame[h_start: h_end, w_start: w_end, :]
                label = efficient_pose.animatePose(temp_img, im0s, track_box, offset,
                                                   segment_width=int(frame.shape[0] / 80),
                                                   marker_radius=int(frame.shape[0] / 80))
                cv2.rectangle(im0s, (int(track_box[0]), int(track_box[1])),
                             (int(track_box[2]), int(track_box[3])), [166, 116, 2], 4)
                cv2.putText(im0s, label, (int(track_box[0]), int(track_box[1]) + 24), 0, 1, [166, 116, 2],
                           thickness=2, lineType=cv.LINE_AA, )
                track.pre_state = track.current_state
                continue
            """
        for *xyxy, conf, cls in yolo_preds:
            img = im0s[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
            label = efficient_pose.estimatePose(img, 'bin.jpg')
            print(label)
            annotator.box_label(xyxy, label, color=(255, 0, 0))

        # save video
        if args.savevid:
            output.write(annotator.im)


if __name__ == '__main__':
    args = parse_args()
    run(args)