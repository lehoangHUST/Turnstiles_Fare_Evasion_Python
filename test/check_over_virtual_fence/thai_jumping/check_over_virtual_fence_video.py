from __future__ import division, print_function, absolute_import
import sys
import ctypes
import os
import shutil
import threading
import cv2 as cv
import numpy as np
sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")

from detections.yolov5.yolov5_trt import YoLov5TRT

from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker

from utils.deep_sort import generate_detections as gdet
from utils.vf_logic_checking import tlbr2tlwh, cropImage, VirtualFence

################################### path to plugin and engine of the customized human detection 
PLUGIN_LIBRARY = r"plugins/human_yolov5/jetson_TX2/libmyplugins.so"
ENGINE_FILE_PATH = r"plugins/human_yolov5/jetson_TX2/human_yolov5s_v5.engine"
ctypes.CDLL(PLUGIN_LIBRARY)

################################### path to a model of traking and initialize a tracker
NN_BUDGET = None
MAX_COSINE_DISTANCE = 0.3
TRACKING_MODEL = r"models/deep_sort/mars-small128.pb"
encoder = gdet.create_box_encoder(TRACKING_MODEL, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric)

################################### declare a instance of VirtualFence
POINTS = ((431, 1092), (747, 929)) # A and B
virtual_fence = VirtualFence(POINTS[0], POINTS[1])

################################### path to folder containing cropped images
if os.path.exists('output/'):
   shutil.rmtree('output/')
os.makedirs('output/')
CROPPED_IMAGES_PATH = r"output/"

################################### path to video demo and capture frames
VIDEO_PATH = r"test/videos/jumping_right_downward_cam3.MOV"
input_cap = cv.VideoCapture(VIDEO_PATH)
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
print("Size: {}".format(size))

# color for tracked objects 
COLOR = [(255, 255, 0), (204, 0, 153), (26, 209, 255), (71, 107, 107)]

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   frame_count = 0
   while input_cap.isOpened():
      is_read, frame = input_cap.read()
      if not is_read:
         break
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= None)

      if len(result_boxes) == 0:
         print ("frame_count: " + str(frame_count) + " : result_boxes is empty!")
         continue

      tlbr2tlwh(result_boxes)
      detect_box = np.copy(result_boxes)
      # 1 Detection g???m (tlwh, conf, feature)
      # frame ?????u v??o ??? ????y l?? ???nh ch??a ???????c x??? l?? (frame sau khi x??? l?? kh??ng b??? thay ?????i), boxes ?????nh d???ng tlwh
      features = encoder(frame, boxes= result_boxes)
      detections = [Detection(box, confidence, feature)
                     for box, confidence, feature 
                     in zip(detect_box, result_scores, features)]
      # Update tracker
      tracker.predict()
      tracker.update(detections)
      # Update current state of each track in tracks
      virtual_fence.updateCurrentStates(tracker.tracks)
      track_count = 0
      for track in tracker.tracks:
         if not track.is_confirmed() or track.time_since_update > 1:
            continue

         if track.pre_state == None:
            track.pre_state = track.current_state
            continue
         if track.pre_state == track.current_state:
            continue
         else:
            track_box = track.to_tlbr()
            temp_img = frame[int(track_box[1]) : int(track_box[3]), int(track_box[0]) : int(track_box[2]),:]
            cv.imwrite(CROPPED_IMAGES_PATH + r"fc_" + str(frame_count) + 
                           r"_tc_" + str(track_count) + r"_ID_" + str(track.track_id) + r".jpg" 
                        ,temp_img)
            print("saved!")
            track.pre_state = track.current_state
            track_count += 1
            continue
      frame_count += 1

finally:
   # destroy the instance
   yolov5_wrapper.destroy()

input_cap.release()
print("OK!")