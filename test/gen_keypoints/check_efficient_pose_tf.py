import sys
import os
import cv2 as cv
import numpy as np
from PIL import Image
sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")


from pose_estimations.efficient_pose.trt_model import EfficientPose

FOLDER_PATH = r"test/images"
MODEL_VARIANT = r"III"

efficient_pose = EfficientPose(folder_path= r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/output"
                  , model_name= MODEL_VARIANT)
for file_name in os.listdir(FOLDER_PATH):
   image = np.array(Image.open(os.path.join(FOLDER_PATH, file_name)))
   print("===============")
   print("file_name: " + file_name)
   print ("Result: " + efficient_pose.estimatePose(image, file_name= file_name))
