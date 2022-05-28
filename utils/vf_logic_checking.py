from math import sqrt
import numpy as np
import cv2 as cv


def tlbr2tlwh(bbox):
    if bbox.ndim < 2:
        bbox = np.atleast_2d(bbox)
    bbox[:, 2:] -= bbox[:, :2]


def cropImage(img, bbox):
    """Crop a part of image with given bbox
   Parameters
   ----------
      img : cv.Mat
         Input image
      bbox : ndarray 
         The bounding box (tlbr format)
   Return
   ----------
      mat : cv.Mat
         a cropped output image  
   """
    return img[bbox[1]: bbox[3], bbox[0]: bbox[2]]


class VirtualFence:

    def __init__(self, A_point, B_point, offset):
        """Initialize a virtual fence between A and B
      Parameters
      ----------
         A_point: a tuple (x, y) is coor of A point
         B_point: a tuple (x, y) is coor of B point
         offset: the offset of virtual fence (pixels)
      Note
      ----------
         H là hình chiếu của M đến AB được xác định bởi công thức dưới đây:
         self.AB_module: Khoảng cách của 2 điểm A và B
         Assume A(x1,y1), B(x2,y2), M(x,y), H is the projection of M with AB line
            MH =  ((x1 - x2)*y - (y1 - y2)*x - (x1*y2 - x2*y1)) / sqrt((x1-x2)^2 + (y1-y2)^2);
            |                    |                                      |---module of AB vec
            |                    |---[cross product of AM vec and AB vec]                                      
            |---MH can get both positive or negative value (positive if AM)
         """
        self.A_point = A_point
        self.B_point = B_point
        self.offset = offset
        self.AB_module = sqrt(((A_point[0] - B_point[0]) ** 2) + ((A_point[1] - B_point[1]) ** 2))
        self.remainder = self.A_point[0] * self.B_point[1] - self.B_point[0] * self.A_point[1]

    def drawLine(self, frame, color):
        cv.line(frame, self.A_point, self.B_point, color, thickness=4)

    def updateCurrentStates(self, tracks):
        """Update current state of each track in a deepsort's track list (on which side of the virtual fence?)
      Parameters
      ----------
         tracks : List[Track] Từng đối tượng track trong frame
            The list of active tracks at the current time step.  
        """

        for track in tracks:
            bbox = track.to_tlbr()
            xy_center = (int((bbox[0] + bbox[2]) / 2),
                         int((bbox[1] + bbox[3]) / 2))
            numerator = (self.A_point[0] - self.B_point[0]) * xy_center[1] - (self.A_point[1] - self.B_point[1]) * \
                        xy_center[0] - self.remainder
            MH_distance = numerator / self.AB_module
            if MH_distance > self.offset:
                track.current_state = 1
                continue
            if MH_distance < -self.offset:
                track.current_state = -1
                continue
            track.current_state = track.pre_state
