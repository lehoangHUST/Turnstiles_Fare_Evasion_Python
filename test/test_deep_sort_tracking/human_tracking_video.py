from __future__ import division, print_function, absolute_import
import sys
import ctypes
import os
import shutil
import threading
import cv2 as cv
import numpy as np
sys.path.append(r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python")
# load modules for detection
from detections.yolov5.yolov5_trt import YoLov5TRT, get_img_path_batches, plot_one_box, inferThread, warmUpThread
# load modules for tracking
from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from utils.deep_sort import generate_detections as gdet
from utils.vf_logic_checking import tlbr2tlwh

# path to plugin and engine of the customized human detection 
PLUGIN_LIBRARY = r"plugins/human_yolov5/jetson_TX2/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5/jetson_TX2/human_yolov5s_v5.engine"
ctypes.CDLL(PLUGIN_LIBRARY)
# path to a model of traking 
NN_BUDGET = None
MAX_COSINE_DISTANCE = 0.3
TRACKING_MODEL = r"models/deep_sort/mars-small128.pb"
results = []
# path to video demo and capture frames
VIDEO_PATH = r"test/videos/jumping_right_downward_cam3.MOV"
input_cap = cv.VideoCapture(VIDEO_PATH)
# create output folder to store the video
if os.path.exists('output/'):
   shutil.rmtree('output/')
os.makedirs('output/')
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# size = (frame_width, frame_height)
size = (frame_height, frame_width) # apply for input video: "test/videos/*cam(x).MOV" 
print("Size: {}".format(size))
output_video = cv.VideoWriter( "output/tracking_demo.avi", cv.VideoWriter_fourcc(*'MJPG'), 20 , size, isColor = True)
# color for tracked objects 
COLOR = [(255, 255, 0), (204, 0, 153), (26, 209, 255), (71, 107, 107)]

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   encoder = gdet.create_box_encoder(TRACKING_MODEL, batch_size=1)
   metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
   tracker = Tracker(metric)
   while input_cap.isOpened():
      is_read, frame = input_cap.read()
      frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
      if not is_read:
         break
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= 1)
      tlbr2tlwh(result_boxes)
      detect_box = np.copy(result_boxes)
      # 1 Detection gồm (tlwh, conf, feature)
      # frame đầu vào ở đây là ảnh chưa được xử lý (frame sau khi xử lý không bị thay đổi), boxes định dạng tlwh
      features = encoder(frame, boxes= result_boxes)
      detections = [Detection(box, confidence, feature)
                     for box, confidence, feature 
                     in zip(detect_box, result_scores, features)]
      # Update tracker.
      tracker.predict()
      tracker.update(detections)

      # track_boxes = []
      for track in tracker.tracks:
         if not track.is_confirmed() or track.time_since_update > 1:
            continue
         track_box = track.to_tlbr()
         cv.rectangle(frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])),
                        COLOR[track.track_id%3], 6)
         cv.putText(frame,"ID: " + str(track.track_id), (int(track_box[0]), int(track_box[1])), 0, 2, (0, 0, 255), 5)

      output_video.write(frame)

   

finally:
   # destroy the instance
   yolov5_wrapper.destroy()
input_cap.release()
output_video.release()
print("OK!")