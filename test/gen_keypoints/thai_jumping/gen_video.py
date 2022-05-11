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

from pose_estimations.efficient_pose.trt_model import EfficientPose

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
POINTS = ((431, 1092), (747, 929)) # A and B for cam 3
# POINTS = ((260, 1099), (810, 1098)) # A and B for cam 2
virtual_fence = VirtualFence(POINTS[0], POINTS[1])

################################### path to folder containing cropped images
if os.path.exists(r'output/'):
   shutil.rmtree(r'output/')
os.makedirs(r'output/')
os.makedirs(r'output/bin_imgs')
CROPPED_IMAGES_PATH = r"output/"
BIN_IMAGES_PATH = r"output/bin_imgs"

################################### path to video demo and capture frames
VIDEO_PATH = r"test/videos/jumping_right_downward_cam3.MOV"
input_cap = cv.VideoCapture(VIDEO_PATH)
frame_width = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# size = (frame_width, frame_height)
size = (frame_height, frame_width) # apply for input video: "test/videos/jumping_right_downward_cam3.MOV" 
print("Size: {}".format(size))
################################### create output folder to store the video
output_video = cv.VideoWriter( "output/jumping_demo.avi", cv.VideoWriter_fourcc(*'MJPG'), 20 , size, isColor = True)

# color for tracked objects 
COLOR = [(255, 255, 0), (204, 0, 153), (26, 209, 255), (71, 107, 107)]

try:   
   # create a YoLov5TRT instance
   yolov5_wrapper = YoLov5TRT(ENGINE_FILE_PATH)
   # create a EfficientPose instance
   efficient_pose = EfficientPose(folder_path= r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/output/bin_imgs", model_name= "III")

   frame_count = 0
   while input_cap.isOpened():
      is_read, frame = input_cap.read()
      if not is_read:
         break
      frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
      drawable_frame = frame.copy()
      result_boxes, result_scores, result_classid, _ = yolov5_wrapper.inferOneImage(frame, drawable= None)

      if len(result_boxes) == 0:
         print ("frame_count: " + str(frame_count) + " : result_boxes is empty!")
         continue

      tlbr2tlwh(result_boxes)
      detect_box = np.copy(result_boxes)
      # 1 Detection gồm (tlwh, conf, feature)
      # frame đầu vào ở đây là ảnh chưa được xử lý (frame sau khi xử lý không bị thay đổi), boxes định dạng tlwh
      features = encoder(frame, boxes= result_boxes)
      detections = [Detection(box, confidence, feature)
                     for box, confidence, feature 
                     in zip(detect_box, result_scores, features)]
      # Update tracker
      tracker.predict()
      tracker.update(detections)

      # Update current state of each track in tracks
      virtual_fence.updateCurrentStates(tracker.tracks)
      virtual_fence.drawLine(drawable_frame, (225, 235, 25))

      for track in tracker.tracks:
         if not track.is_confirmed() or track.time_since_update > 1:
            continue
         

         # draw confirmed tracks 
         track_box = track.to_tlbr()
         cv.rectangle(drawable_frame, (int(track_box[0]), int(track_box[1])), (int(track_box[2]), int(track_box[3])),
                        COLOR[track.track_id%3], 6)
         cv.putText(drawable_frame,"ID: " + str(track.track_id), (int(track_box[0]), int(track_box[1])), 0, 2, (0, 0, 255), 5)



         if track.pre_state == None:
            track.pre_state = track.current_state
            continue
         if track.pre_state == track.current_state:
            continue
         else:
            
            offset = 20
            h_start = max(0, int(track_box[1]) - offset)
            h_end = min(int(track_box[3]) + offset, frame.shape[0])
            w_start = max(0, int(track_box[0]) - offset)
            w_end =  min(int(track_box[2]) + offset, frame.shape[1]) 

            temp_img = frame[h_start : h_end, w_start : w_end, :]
            # temp_img = frame[int(track_box[1]) : int(track_box[3]), int(track_box[0]) : int(track_box[2]),:]

            label = efficient_pose.estimatePose(np.array(temp_img[...]), file_name= r"fc_{0}_tc_{1}.png".format(frame_count, track.track_id), stored= True)
            cv.putText(drawable_frame, label , (270, 270), 0, 6, [0, 0, 250], thickness= 10, lineType= cv.LINE_AA,)

            track.pre_state = track.current_state
            continue
      output_video.write(drawable_frame)
      frame_count += 1

finally:
   # destroy the instance
   yolov5_wrapper.destroy()
   efficient_pose.destroy()

input_cap.release()
output_video.release()
print("OK!")