import cv2 as cv

IMAGE_PATH = r"/home/thaivu/Projects/TestImages/test2_4human.jpg"
WRITING_PATH = r"/home/thaivu/Projects/Turnstiles_Fare_Evasion_Python/output/crop_test.jpg"

img = cv.imread(IMAGE_PATH)
print(img.shape)
img = img[12:200, 30:1000]
cv.imwrite(WRITING_PATH, img)