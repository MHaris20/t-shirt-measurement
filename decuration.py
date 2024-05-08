import cv2
import numpy as np


def draw_circle(image, corrdinate, radius=20, color=(255, 0, 0), thickness=3):
    return cv2.circle(image, corrdinate, radius, color, thickness)


def draw_two_sided_arrow(image, start, end, color, thickness, tipLength):
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=tipLength)
    cv2.arrowedLine(image, end, start, color, thickness, tipLength=tipLength)


def draw_horizontal_dash_line(image, y1, color, thickness, dash_length):
    h, w = image.shape[:2]
    
    # Dash line
    x, y = 0, y1
    while x < w:
        x_end = x + dash_length
        x_end = min(x_end, w)
        cv2.line(image, (x, y), (x_end, y), color, thickness)
        x = x_end + dash_length


def draw_vertical_dash_line(image, x1, color, thickness, dash_length):
    h, w = image.shape[:2]
    
    # Dash line
    x, y = x1, 0
    while y < h:
        y_end = y + dash_length
        y_end = min(y_end, h)
        cv2.line(image, (x, y), (x, y_end), color, thickness)
        y = y_end + dash_length


def draw_dashed_rectangle(image, start_point, end_point, color, thickness, dash_length):
    h, w = image.shape[:2]
    x1, y1 = start_point
    x2, y2 = end_point

    # Top line
    x, y = 0, y1
    while x < w:
        x_end = x + dash_length
        x_end = min(x_end, w)
        cv2.line(image, (x, y), (x_end, y), color, thickness)
        x = x_end + dash_length

    # Right line
    x, y = x2, 0
    while y < h:
        y_end = y + dash_length
        y_end = min(y_end, h)
        cv2.line(image, (x, y), (x, y_end), color, thickness)
        y = y_end + dash_length

    # Bottom line
    x, y = w, y2
    while x > 0:
        x_end = x - dash_length
        x_end = max(x_end, 0)
        cv2.line(image, (x, y), (x_end, y), color, thickness)
        x = x_end - dash_length

    # Left line
    x, y = x1, h
    while y > 0:
        y_end = y - dash_length
        y_end = max(y_end, 0)
        cv2.line(image, (x, y), (x, y_end), color, thickness)
        y = y_end - dash_length


def custom_scale(frame):
    h, w = frame.shape[:2]

    ### Vertical scale background
    frame = cv2.rectangle(frame, (0, 0), (148, h), (178, 86, 13), -1)

    ### Vertical scale small line
    inch_v = 0
    cm_distance = 16
    total_cm = int(h/cm_distance)
    for cm in range(0, total_cm):
        if cm%10==0:
            frame = cv2.line(frame, (148, h-int((cm*cm_distance)+148)), (50, h-int((cm*cm_distance)+148)), (255, 255, 255), 3)
            frame = cv2.putText(frame, str(inch_v), (15, h-int((cm*cm_distance)+148)-25), cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 255, 255), 2, cv2.LINE_AA)
            inch_v+=1
        else:    
            frame = cv2.line(frame, (148, h-int((cm*cm_distance)+148)), (100, h-int((cm*cm_distance)+148)), (255, 255, 255), 3)


    ### Horizontal scale background
    frame = cv2.rectangle(frame, (148, h-150), (w, h), (178, 86, 13), -1)

    ### Horizontal scale small line
    inch_h = 0
    cm_distance = 16
    total_cm = int((w-148)/cm_distance)
    for cm in range(0, total_cm):
        if cm%10==0:
            frame = cv2.line(frame, (int((cm*cm_distance)+148), h-148), (int((cm*cm_distance)+148), h-50), (255, 255, 255), 3)
            frame = cv2.putText(frame, str(inch_h), (int((cm*cm_distance)+148)+10, h-40), cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 255, 255), 2, cv2.LINE_AA)
            inch_h+=1
        else:    
            frame = cv2.line(frame, (int((cm*cm_distance)+148), h-148), (int((cm*cm_distance)+148), h-100), (255, 255, 255), 3)

    return frame

        
def hint(img, text):
    # Define text and box properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    box_color = (0, 0, 255)
    box_thickness = 2
    box_padding = 10
    box_coords = ((50, 50), (300, 200))  # top-left and bottom-right coordinates of the box

    # Calculate arrow coordinates
    box_center = ((box_coords[0][0] + box_coords[1][0]) // 2, (box_coords[0][1] + box_coords[1][1]) // 2)
    arrow_length = 100
    arrow_start = (box_center[0], box_coords[1][1] + box_padding)
    arrow_end = (box_center[0], box_coords[1][1] + box_padding + arrow_length)

    # Draw box and arrow
    cv2.rectangle(img, box_coords[0], box_coords[1], box_color, box_thickness)
    cv2.arrowedLine(img, arrow_start, arrow_end, box_color, box_thickness)

    # Draw text
    text_coords = (box_center[0] - text_size[0] // 2, box_coords[1][1] + box_padding + arrow_length + text_size[1])
    cv2.putText(img, text, text_coords, font, font_scale, box_color, thickness, cv2.LINE_AA)
