# @dataset_brief
# - each image in a sperable folder will be generated to coordinates of human's body join (x1, y1, x2, y2, ...)
# - save the coordinates to pandas table with labels (determined based on folder name)
# - split the obtained dataset to training, testing, validating set (75%, 20%, 5% respectively ?) 

import os
import sys
import cv2 as cv
import csv
import shutil

sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")
from pose_estimations.efficient_pose.trt_model import EfficientPose

WALKING_FOLDER_PATH = R"/home/thaivu169/Pictures/SVM_action_classification/walking"
NOT_WALKING_FOLDER_PATH = R"/home/thaivu169/Pictures/SVM_action_classification/not_walking"

# make new output folder contained svm-dataset's results
if os.path.exists(r'output/svm_dataset'):
   shutil.rmtree(r'output/svm_dataset')
os.makedirs(r'output/svm_dataset')
if os.path.exists(r'output/svm_dataset/svm_bin_imgs'):
   shutil.rmtree(r'output/svm_dataset/svm_bin_imgs')
os.makedirs(r'output/svm_dataset/svm_bin_imgs')

# paths to output results
BIN_IMGS_PATH = r'output/svm_dataset/svm_bin_imgs'
DATASET_RESULT_PATH = r"output/svm_dataset/svm_dataset.csv" 

# writing the fields to the .csv file
FIELDS = ['head_top_x', 'head_top_y', 'upper_neck_x', 'upper_neck_y', 
        'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x', 'right_elbow_y', 
        'right_wrist_x', 'right_wrist_y', 'thorax_x', 'thorax_y',
        'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x', 'left_elbow_y',
        'left_wrist_x', 'left_wrist_y', 'pelvis_x', 'pelvis_y',
        'right_hip_x', 'right_hip_y', 'right_knee_x', 'right_knee_y',
        'right_ankle_x', 'right_ankle_y', 'left_hip_x', 'left_hip_y',
        'left_knee_x', 'left_knee_y', 'left_ankle_x', 'left_ankle_y', "labels"] # label = 1 (walking), = 0 (not walking)
LABELS = {"walking" : 1, "not_walking" : 0}
with open(DATASET_RESULT_PATH, "a", newline= '') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(FIELDS)

try:
    print("initializing EfficientPose model!")
    efficient_pose = EfficientPose(folder_path= BIN_IMGS_PATH, model_name= "RT")
    print("===initialize EfficientPose model successfully!")
    print("===running in '" + WALKING_FOLDER_PATH + "' folder")
    for file_name in os.listdir(WALKING_FOLDER_PATH):
        temp_img = cv.imread(os.path.join(WALKING_FOLDER_PATH, file_name)) # rgb image
        temp_list = []
        coordinates = efficient_pose.genCoordinates(temp_img, file_name)
        if coordinates is None:
            print ("existing coordinates is none!")
            continue
        # save keypoints's coordinates to csv writer object
        for idx in range(16):
            _, keypts_x, keypts_y = coordinates[idx] # coordinates was normalized (0 -> 1)
            temp_list.append(keypts_x)
            temp_list.append(keypts_y)
        # append label for the data row
        temp_list.append(LABELS['walking'])
        with open(DATASET_RESULT_PATH, "a", newline= '') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(temp_list)

    print("===running in '" + NOT_WALKING_FOLDER_PATH + "' folder")
    for file_name in os.listdir(NOT_WALKING_FOLDER_PATH):
        temp_img = cv.imread(os.path.join(NOT_WALKING_FOLDER_PATH, file_name)) # rgb image
        temp_list = []
        coordinates = efficient_pose.genCoordinates(temp_img, file_name)
        if coordinates is None:
            print ("existing coordinates is none!")
            continue
        # save keypoints's coordinates to csv writer object
        for idx in range(16):
            _, keypts_x, keypts_y = coordinates[idx] # coordinates was normalized (0 -> 1)
            temp_list.append(keypts_x)
            temp_list.append(keypts_y)
        # append label for the data row
        temp_list.append(LABELS['not_walking'])
        with open(DATASET_RESULT_PATH, "a", newline= '') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(temp_list)

finally:
    # destroy the instance
    efficient_pose.destroy()
