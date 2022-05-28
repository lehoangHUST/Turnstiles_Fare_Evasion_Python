import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import cv2 as cv
import time
import threading

from utils.efficient_pose import helpers

RESOLUTIONS_DICT = {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256,
                    'ii_lite': 368}
POSE_DICT = {0: 'walking', 1: 'jumping'}


class EfficientPose:
    def __init__(self, folder_path, model_name="III"):
        # path to store images (after generating keypoints)
        self.folder_path = folder_path
        self.model_variant = model_name.lower()
        assert (self.model_variant in ['efficientposert', 'rt', 'efficientposei', 'i', 'efficientposeii', 'ii',
                                       'efficientposeiii', 'iii', 'efficientposeiv', 'iv', 'efficientposert_lite',
                                       'rt_lite', 'efficientposei_lite', 'i_lite', 'efficientposeii_lite', 'ii_lite'])
        self.model_variant = self.model_variant[13:] if len(self.model_variant) > 7 else self.model_variant
        self.resolution = RESOLUTIONS_DICT[self.model_variant]
        self.lite = True if self.model_variant.endswith('_lite') else False

        # ==================== Tensorrt ====================
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(os.path.join("plugins", "efficient_pose", "jetson_TX2",
                               "EfficientPose{0}.plan".format(self.model_variant.upper())), "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                # self.input_w = engine.get_binding_shape(binding)[-1]
                # self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, batch):
        """
      Perform inference on supplied image batch.
      Args:
         batch: ndarray
            Stack of preprocessed images
      Returns:
         EfficientPose model outputs for the supplied batch.
      """
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        batch_input_image = np.empty(shape=[self.batch_size, self.resolution, self.resolution, 3])
        np.copyto(batch_input_image[0], batch)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        batch_outputs = host_outputs[0]
        return np.reshape(batch_outputs, (1, self.resolution, self.resolution, 16))

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it
        self.ctx.pop()

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
        batch = np.expand_dims(image, axis=0)  # mini-batch = 1
        # Preprocess batch
        batch = helpers.preprocess(batch, self.resolution, self.lite)
        # Perform inference
        batch_outputs = self.infer(batch)
        # Extract coordinates
        coordinates = helpers.extract_coordinates(batch_outputs[0, ...], image_height, image_width)
        # Print processing time
        print(
            '\n##########################################################################################################')
        print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
        print(
            '##########################################################################################################\n')
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
        image = Image.new(mode="1", size=(image_width, image_height))  # default: image's color is black
        # Annotate image
        image_draw = ImageDraw.Draw(image)
        helpers.drawBinImage(image_draw, coordinates, image_height, image_width, segment_width=int(image_side / 50),
                             marker_radius=int(image_side / 100))
        # Save annotated image
        image.save(file_name)

    def genCoordinates(self, image, file_name, stored=True):
        """
      Process of generating keypoints's coordinates in the image.
      Args:
         image: ndarray
            The image to analyze
         file_name: string
            The file's name to store binary image
         stored: boolean
            Flag to store visualization of predicted poses
      Returns:
         keypoints's coordinates in the image
      """
        # perform inference
        coordinates = self.analyzeOneImage(image)

        if stored and self.folder_path and coordinates is not None:
            image_height = image.shape[0]
            image_width = image.shape[1]
            self.saveBinImage2Folder(os.path.join(self.folder_path, file_name), coordinates, image_height, image_width)
            print("save bin image with file name: " + file_name + " successfully!")
        return coordinates

    def estimatePose(self, image, file_name, stored=True):
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
        v_left_femur = helpers.genVector2D(coordinates[13][1:], coordinates[14][1:])

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

    def animatePose(self, image, drawable_image, track_box, offset=20, segment_width=5, marker_radius=5):
        """
      animate the human pose to drawable_image based on image (contains only a persion)
      Args:
         image: ndarray
            The image to analyze (ảnh trích từ các track qua hàng rào ảo)
         file_name: ndarray
            The animated image (có kích thước lớn hơn image, ảnh frame)
      """
        # perform inference
        coordinates = self.analyzeOneImage(image)
        # body_part_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
        body_part_colors = (
        (12, 53, 200), (54, 100, 32), (200, 165, 23), (212, 2, 100), (105, 200, 50), (50, 200, 205), (230, 23, 23),
        (56, 67, 87), (54, 123, 123), (50, 200, 205), (212, 76, 54), (200, 165, 23), (98, 98, 98), (130, 103, 130),
        (160, 189, 76), (133, 21, 51))

        image_height = image.shape[0]
        image_width = image.shape[1]
        # Draw markers
        for i, (_, body_part_x, body_part_y) in enumerate(coordinates):
            body_part_x *= image_width
            body_part_x = body_part_x - offset + track_box[0]

            body_part_y *= image_height
            body_part_y = body_part_y - offset + track_box[1]
            cv.circle(drawable_image, (int(body_part_x), int(body_part_y)), radius=marker_radius,
                      color=body_part_colors[i], thickness=-1)

        # Define segments and colors
        segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11),
                    (11, 12), (13, 14), (14, 15)]
        segment_colors = body_part_colors
        # Draw segments
        for (body_part_a_index, body_part_b_index) in segments:
            _, body_part_a_x, body_part_a_y = coordinates[body_part_a_index]
            body_part_a_x *= image_width
            body_part_a_x = body_part_a_x - offset + track_box[0]
            body_part_a_y *= image_height
            body_part_a_y = body_part_a_y - offset + track_box[1]
            _, body_part_b_x, body_part_b_y = coordinates[body_part_b_index]
            body_part_b_x *= image_width
            body_part_b_x = body_part_b_x - offset + track_box[0]
            body_part_b_y *= image_height
            body_part_b_y = body_part_b_y - offset + track_box[1]
            cv.line(drawable_image, (int(body_part_a_x), int(body_part_a_y)), (int(body_part_b_x), int(body_part_b_y)),
                    color=segment_colors[body_part_b_index], thickness=segment_width)

        v_backbone = helpers.genVector2D(coordinates[9][1:], coordinates[5][1:])
        v_right_femur = helpers.genVector2D(coordinates[10][1:], coordinates[11][1:])
        v_left_femur = helpers.genVector2D(coordinates[13][1:], coordinates[14][1:])
        bb_rf_angle = helpers.calcAngle2Vector(v_backbone, v_right_femur)
        bb_lf_angle = helpers.calcAngle2Vector(v_backbone, v_left_femur)

        print("The angle between v_backbone and v_right_femur: " + str(bb_rf_angle))
        print("The angle between v_backbone and v_left_femur: " + str(bb_lf_angle))

        if (bb_rf_angle > 90 and bb_lf_angle > 90):
            return POSE_DICT[0]
        else:
            return POSE_DICT[1]
