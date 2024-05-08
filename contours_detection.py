import cv2
import numpy as np
from helper_functions import get_bounding_box

def detect_contours(image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE, threshold=0.0004):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(gray, mode, method)
    image = np.full((h, w, 3), 255, dtype = np.uint8)

    x1y1, x2y2, x3y3, x4y4 = 0, 0, 0, 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        epsilon = threshold * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)
        
        if (cv2.contourArea(c)) > 10:
            x1y1 = (x, y)
            x2y2 = (x+w, y)
            x3y3 = (x+w, y+h)
            x4y4 = (x, y+h)

    return approx, (x1y1, x2y2, x3y3, x4y4), image


def detect_min_max_contours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, threshold=0.0004):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(gray, mode, method)

    final_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            final_contours.append(contour)

    min_area = min(final_contours, key=cv2.contourArea)
    max_area = max(final_contours, key=cv2.contourArea)

    epsilon = threshold * cv2.arcLength(max_area, True)
    max_approx = cv2.approxPolyDP(max_area, epsilon, True)

    return (min_area, get_bounding_box(min_area)), (max_area, get_bounding_box(max_area)), max_approx


def detect_approx_contours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, threshold=0.0004):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    contours, _ = cv2.findContours(gray, mode, method)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            epsilon = threshold * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return approx
    return None
    
    
