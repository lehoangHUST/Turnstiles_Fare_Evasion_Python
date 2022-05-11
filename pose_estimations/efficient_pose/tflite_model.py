import os
import numpy as np

from utils.efficient_pose import helpers

from tensorflow.python.platform.gfile import FastGFile
from tensorflow.compat.v1 import GraphDef
from tensorflow.compat.v1.keras.backend import get_session
from tensorflow import import_graph_def

from tensorflow import lite

import time

RESOLUTIONS_DICT = {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}
POSE_DICT = {0: 'walking', 1: 'jumping'}
class EfficientPose:
   def __init__(self, folder_path, model_name= "IV"):
      # path to store images (after generating keypoints)
      self.folder_path = folder_path
      self.model_variant = model_name.lower()
      assert (self.model_variant in ['efficientposert', 'rt', 'efficientposei', 'i', 'efficientposeii', 'ii', 'efficientposeiii', 'iii', 'efficientposeiv', 'iv', 'efficientposert_lite', 'rt_lite', 'efficientposei_lite', 'i_lite', 'efficientposeii_lite', 'ii_lite'])
      self.model_variant = self.model_variant[13:] if len(self.model_variant) > 7 else self.model_variant 
      self.resolution = RESOLUTIONS_DICT[self.model_variant]
      self.lite = True if self.model_variant.endswith('_lite') else False
      
      # load model
      self.model = lite.Interpreter(model_path= os.path.join('models', 'efficient_pose', 'tflite', 'EfficientPose{0}.tflite'.format(self.model_variant.upper())))
      self.model.allocate_tensors()

   def infer(self, batch):
      """
      Perform inference on supplied image batch.
      Args:
         batch: ndarray
            Stack of preprocessed images
      Returns:
         EfficientPose model outputs for the supplied batch.
      """
      input_details = self.model.get_input_details()
      output_details = self.model.get_output_details()
      self.model.set_tensor(input_details[0]['index'], batch)
      self.model.invoke()
      batch_outputs = self.model.get_tensor(output_details[-1]['index'])
      return batch_outputs

   def analyzeOneImage(self, image):
      """
      Predict pose coordinates on supplied image.
      Args:
         image: ndarray
            The image to analyze
      Returns:
         Predicted pose coordinates in the supplied image.
      """
      start_time = time.time()
      image_height, image_width = image.shape[:2]
      batch = np.expand_dims(image, axis=0) # mini-batch = 1
      # Preprocess batch
      batch = helpers.preprocess(batch, self.resolution, self.lite)
      # Perform inference
      batch_outputs = self.infer(batch)
      # Extract coordinates
      coordinates = helpers.extract_coordinates(batch_outputs[0,...], image_height, image_width)
      # Print processing time
      print('\n##########################################################################################################')
      print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
      print('##########################################################################################################\n')
      return coordinates
   
   def saveBinImage2Folder(self, file_name, coordinates, image_height, image_width):
      """
      Annotates supplied image from predicted coordinates.
      Args:
         file_name: string
            The path to folder to store binary image
         coordinates: list
            Predicted body part coordinates for image, image's height, image's with
      """
      # Load raw image
      from PIL import Image, ImageDraw
      image_side = image_width if image_width >= image_height else image_height
      image = Image.new(mode= "1", size= (image_width, image_height)) # default: image's color is black
      # Annotate image
      image_draw = ImageDraw.Draw(image)
      helpers.drawBinImage(image_draw, coordinates, image_height, image_width, segment_width= int(image_side/50), marker_radius= int(image_side/100))
      # Save annotated image
      image.save(file_name)
   
   def performPoseEstimation(self, image, file_name, stored= True):
      """
      Process of estimating poses frome image.
      Args:
         image: ndarray
            The image to analyze
         file_name: string
            The file's name to store binary image
         stored: boolean
            Flag to store visualization of predicted poses
      Returns:
         Boolean expressing if tracking was successfully performed.
      """
      # perform inference
      coordinates = self.analyzeOneImage(image)
      
      if stored and self.folder_path and coordinates is not None:
         image_height = image.shape[0]
         image_width = image.shape[1]
         self.saveBinImage2Folder(os.path.join(self.folder_path, file_name), coordinates, image_height, image_width)
         print("save the bin image successfully!")
      return True

   def estimatePose(self, image, file_name, stored= True):
      """
      Estimate of the human pose from image (contains only a persion)
      Args:
         image: ndarray
            The image to analyze
         file_name: string
            The file's name to store binary image
         stored: boolean
            Flag to store visualization of predicted poses
      Returns:
         String expressing the human pose in the image
      """
      # perform inference
      coordinates = self.analyzeOneImage(image)
      v_backbone = helpers.genVector2D(coordinates[9][1:], coordinates[5][1:])
      print(v_backbone)
      v_right_femur = helpers.genVector2D(coordinates[10][1:], coordinates[11][1:])
      v_left_femur =  helpers.genVector2D(coordinates[13][1:], coordinates[14][1:])
      
      bb_rf_angle = helpers.calcAngle2Vector(v_backbone, v_right_femur)
      bb_lf_angle = helpers.calcAngle2Vector(v_backbone, v_left_femur)

      print("The angle between v_backbone and v_right_femur: " + str(bb_rf_angle))
      print("The angle between v_backbone and v_left_femur: " + str(bb_lf_angle))
      
      if stored and self.folder_path and coordinates is not None:
         image_height = image.shape[0]
         image_width = image.shape[1]
         self.saveBinImage2Folder(os.path.join(self.folder_path, file_name), coordinates, image_height, image_width)
         print("save the bin image successfully!")

      if (bb_rf_angle > 90 and bb_lf_angle > 90):
         return POSE_DICT[0]
      else:
         return POSE_DICT[1]