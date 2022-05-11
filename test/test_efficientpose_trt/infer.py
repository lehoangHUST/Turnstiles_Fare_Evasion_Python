# Refer: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_model_python
# Refer: https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
import numpy as np
import cv2 as cv
sys.path.append(r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python")
from utils.efficient_pose import helpers
from PIL import Image

TRT_LOGGER = trt.Logger()
ENGINE_FILE_PATH = r"plugins/efficient_pose/jetson_TX2/engine.plan"


def load_engine(engine_file_path):
   assert os.path.exists(engine_file_path)
   print("Reading engine from file {}".format(engine_file_path))
   with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
      return runtime.deserialize_cuda_engine(f.read())


def infer(engine, input_file):
   print("Reading input image from file {}".format(input_file))
   img = cv.imread(input_file) 
   image_height, image_width = img.shape[:2]
   input_image = np.expand_dims(img, axis=0) # mini-batch = 1
   # Preprocess batch
   input_image = helpers.preprocess(input_image, 480, False)

   with engine.create_execution_context() as context:
      ctx = cuda.Device(0).make_context()

      # Set input shape based on image dimensions for inference
      context.set_binding_shape(engine.get_binding_index("input_res1:0"), (1, 480, 480, 3))
      # Allocate host and device buffers
      bindings = []
      for binding in engine:
         binding_idx = engine.get_binding_index(binding)
         size = trt.volume(context.get_binding_shape(binding_idx))
         dtype = trt.nptype(engine.get_binding_dtype(binding))
         if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(input_image)
            input_memory = cuda.mem_alloc(input_image.nbytes)
            bindings.append(int(input_memory))
         else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

      ctx.push()
      stream = cuda.Stream()
      # Transfer input data to the GPU.
      cuda.memcpy_htod_async(input_memory, input_buffer, stream)
      # Run inference
      context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
      # Transfer prediction output from the GPU.
      cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
      # Synchronize the stream
      stream.synchronize()

   new_output = np.reshape(output_buffer, (1,480,480,16))
   print(new_output.shape)
   coordinates = helpers.extract_coordinates(new_output[0,...], image_height, image_width)
   v_backbone = helpers.genVector2D(coordinates[9][1:], coordinates[5][1:])
   print(v_backbone)
   v_right_femur = helpers.genVector2D(coordinates[10][1:], coordinates[11][1:])
   v_left_femur =  helpers.genVector2D(coordinates[13][1:], coordinates[14][1:])
   
   bb_rf_angle = helpers.calcAngle2Vector(v_backbone, v_right_femur)
   bb_lf_angle = helpers.calcAngle2Vector(v_backbone, v_left_femur)

   print("The angle between v_backbone and v_right_femur: " + str(bb_rf_angle))
   print("The angle between v_backbone and v_left_femur: " + str(bb_lf_angle))

   if (bb_rf_angle > 90 and bb_lf_angle > 90):
      print("walking")
   else:
      print("jumping")

   ctx.pop()

print("Running TensorRT inference ...")

FOLDER_PATH = r"test/images"
for file_name in os.listdir(FOLDER_PATH):
   print("===============")
   print("file_name: " + file_name)
   engine = load_engine(r"plugins/efficient_pose/jetson_TX2/EfficientPoseIII.plan") 
   infer(engine, os.path.join(FOLDER_PATH, file_name))


   # with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
   #    print("Writing output image to file {}".format(output_file))
   #    img.convert('RGB').save(output_file, "PPM")

