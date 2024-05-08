import cv2
from measurement import Measurement


measure = Measurement()


image = cv2.imread('images/image.png')
measure.detection(image)