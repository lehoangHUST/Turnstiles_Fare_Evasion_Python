# Refer: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_model_python
# Refer: https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
import numpy as np
import cv2 as cv
import shutil
sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")
from pose_estimations.efficient_pose.trt_model import EfficientPose


################################### path to folder containing cropped images
if os.path.exists(r'output/'):
   shutil.rmtree(r'output/')
os.makedirs(r'output/')
os.makedirs(r'output/bin_imgs')
CROPPED_IMAGES_PATH = r"output/"
BIN_IMAGES_PATH = r"output/bin_imgs"

try:
   efficient_pose = EfficientPose(folder_path= r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/output/bin_imgs")

   FOLDER_PATH = r"test/images"
   for file_name in os.listdir(FOLDER_PATH):
      print("===============")
      print("file_name: " + file_name)
      img = cv.imread(os.path.join(FOLDER_PATH, file_name))
      efficient_pose.estimatePose(img, file_name)
      

finally:
   efficient_pose.destroy()
print("OK!")



   # with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
   #    print("Writing output image to file {}".format(output_file))
   #    img.convert('RGB').save(output_file, "PPM")

