import cv2
import time
import numpy as np
from global_variables import coin_mm, inche_per_mm
from contours_detection import detect_contours, detect_min_max_contours, detect_approx_contours
from image_preprocessing import mask_method_1, mask_method_2, mask_method_3, erotion, canny_edge, dilation
from helper_functions import get_intersection, get_area_by_pct, split_by_y, center_value, get_center_point_by_x, split_by_x, show
from decuration import hint, draw_dashed_rectangle, draw_two_sided_arrow, draw_horizontal_dash_line, draw_vertical_dash_line, draw_circle, custom_scale

class Measurement:
    def __init__(self) -> None:
        pass


    def coin_length_pixels(self, image, min_bbox):
        y1 = min_bbox[0][1]
        y3 = min_bbox[2][1]
        length_pixels = abs(y3-y1)
        self.coin_per_mm = coin_mm/length_pixels


    def sleeve_length(self, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox
        h, w = image.shape[:2]

        contours = [(c[0][0], c[0][1]) for c in max_approx]
        self.left_sleeve_top, self.right_sleeve_top = split_by_x(contours)
        
        bottom_top_sleeve = []
        for c in max_approx:
            x = c[0][0]
            y = c[0][1]

            if y > self.left_sleeve_top[1] and y > self.right_sleeve_top[1]:
                bottom_top_sleeve.append((x, y))

        self.left_sleeve_bottom, self.right_sleeve_bottom = split_by_x(bottom_top_sleeve)

        draw_two_sided_arrow(image, self.left_sleeve_top, self.left_sleeve_bottom, (255, 0, 255), 9, 0.09)
        draw_two_sided_arrow(image, self.right_sleeve_top, self.right_sleeve_bottom, (255, 0, 255), 9, 0.09)

        length = np.linalg.norm(np.array([self.left_sleeve_top]) - np.array([self.left_sleeve_bottom]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Sleeves Length Left : {length:.2f} Inch"), (w-800, 70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)

        length = np.linalg.norm(np.array([self.right_sleeve_top]) - np.array([self.right_sleeve_bottom]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Sleeves Length Right : {length:.2f} Inch"), (w-800, 140), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)

    
    def width_top(self, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox
        thresh_left = get_area_by_pct(self.left_sleeve_top[1], self.left_sleeve_bottom[1], 40)
        
        left_sleeve = []
        for c in max_approx:
            x = c[0][0]
            y = c[0][1]

            if (x > self.left_sleeve_bottom[0] and y > self.left_sleeve_top[1]) and (x < self.right_sleeve_bottom[0] and y > self.right_sleeve_top[1]) and (y < self.left_sleeve_bottom[1]+thresh_left):
                left_sleeve.append((x, y))

        self.left_width_top, self.right_width_top = split_by_x(left_sleeve)

        length = np.linalg.norm(np.array([self.left_width_top]) - np.array([self.right_width_top]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Width Top : {length:.2f} Inch"), (30, 70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)

        draw_two_sided_arrow(image, self.left_width_top, self.right_width_top, (255, 0, 255), 6, 0.05)


    def width_mid(self, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox
        thresh_left = get_area_by_pct(self.left_sleeve_top[1], self.left_sleeve_bottom[1], 40)
        
        left_sleeve = []
        for c in max_approx:
            x = c[0][0]
            y = c[0][1]

            if y > self.left_sleeve_bottom[1]+thresh_left:
                left_sleeve.append((x, y))

        self.left_width_bottom, self.right_width_bottom = split_by_x(left_sleeve)

        thresh_center_width = get_area_by_pct(self.left_width_top[1], self.left_width_bottom[1], 10)
        center_width_y = center_value(self.left_width_top[1], self.left_width_bottom[1])

        center_width = []
        for c in max_contours:
            x = c[0][0]
            y = c[0][1]

            if y in range(center_width_y, center_width_y+thresh_center_width):
                center_width.append((x, y))

        self.left_width_center, self.right_width_center = split_by_x(center_width)

        length = np.linalg.norm(np.array([self.left_width_center]) - np.array([self.right_width_center]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Width Center : {length:.2f} Inch"), (30, 140), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)

        draw_two_sided_arrow(image, self.left_width_center, self.right_width_center, (255, 0, 255), 6, 0.05)

    
    def length_right(self, orignal_image, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox

        thresh_bottom = get_area_by_pct(y1, y3, 10)
        crop = mask_method_2(orignal_image[:y1+thresh_bottom,self.left_width_top[0]:self.right_width_top[0]])
        crop_approx = detect_approx_contours(crop, threshold=0.01)
        thresh_crop = get_area_by_pct(0, y1+thresh_bottom, 10)

        self.mid_collar, _ = split_by_y([(appr[0][0], appr[0][1]) for appr in crop_approx])

        crop_coords = []
        for appr in crop_approx:
            x = appr[0][0]
            y = appr[0][1]
            
            if x > self.mid_collar[0] and y in range(0, abs((y1+thresh_bottom)-thresh_crop)):
                crop_coords.append((x, y))


        _, self.right_length_top = split_by_y(crop_coords)

        right_x = (x1-(x1-self.left_width_top[0]))+self.right_length_top[0]
        self.right_length_top = (right_x, self.right_length_top[1])
        draw_circle(image, self.right_length_top , color=(0, 255, 0))

        thresh_right = get_area_by_pct(self.right_length_top[1], self.right_sleeve_top[1], 10)
        
        length_bottom_coords = []
        for c in max_contours:
            x = c[0][0]
            y = c[0][1]
            
            if x in range(self.right_length_top[0], self.right_length_top[0]+thresh_right):
                length_bottom_coords.append((x, y))

        _, self.right_length_bottom = split_by_y(length_bottom_coords)
        draw_circle(image, self.right_length_bottom , color=(0, 255, 0))

        length = np.linalg.norm(np.array([self.right_length_top]) - np.array([self.right_length_bottom]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Length : {length:.2f} Inch"), (30, 210), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)
        
        draw_two_sided_arrow(image, self.right_length_top, self.right_length_bottom, (255, 0, 255), 6, 0.05)


    def sleeve_width_left(self, orignal_image, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox
        h, w = image.shape[:2]
        
        mask = mask_method_3(orignal_image).copy()
        
        thresh_bottom = get_area_by_pct(self.left_sleeve_top[1], self.left_sleeve_bottom[1], 5)
        thresh_right = get_area_by_pct(self.left_sleeve_top[1], self.left_sleeve_bottom[1], 50)
        thresh_line = get_area_by_pct(self.left_sleeve_top[1], self.left_sleeve_bottom[1], 10)
        
        
        crop = mask[self.left_sleeve_top[1]:self.left_sleeve_top[1]+thresh_bottom, self.left_width_top[0]:self.left_width_top[0]+thresh_right]
        crop_approx = detect_approx_contours(crop, threshold=0.01)
        width_left_coords = [(c[0][0], c[0][1]) for c in crop_approx]
        min_coord, _ = split_by_x(width_left_coords)
        
        left_x = (x1-(x1-self.left_width_top[0]))+min_coord[0]
        left_y = (y1-(y1-self.left_sleeve_top[1]))+min_coord[1]
        self.left_sleeve_width = (left_x, left_y)
        draw_circle(image, self.left_sleeve_width , color=(0, 0, 255))
        
        crop = erotion(mask[:self.left_sleeve_width[1], self.left_sleeve_width[0]-thresh_line:self.left_sleeve_width[0]])
        crop_approx = detect_approx_contours(crop, threshold=0.01)
        width_left_coords = [(c[0][0], c[0][1]) for c in crop_approx]
        min_coord, _ = split_by_y(width_left_coords)
        
        left_x = (x1-(x1-(self.left_sleeve_width[0]-thresh_line)))+min_coord[0]
        self.left_sleeve_width = (left_x, min_coord[1])
        
        draw_circle(image, self.left_sleeve_width , color=(0, 255, 0))
        
        length = np.linalg.norm(np.array([self.left_sleeve_top]) - np.array([self.left_sleeve_width]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Sleeve left Width : {length:.2f} Inch"), (w-800, 210), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)
        
        draw_two_sided_arrow(image, self.left_sleeve_top, self.left_sleeve_width, (255, 0, 255), 6, 0.05)
        
        
    def sleeve_width_right(self, orignal_image, image, max_contours, max_bbox, max_approx):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = max_bbox
        h, w = image.shape[:2]
        
        mask = mask_method_3(orignal_image).copy()
        
        thresh_bottom = get_area_by_pct(self.right_sleeve_top[1], self.right_sleeve_bottom[1], 20)
        thresh_right = get_area_by_pct(self.right_sleeve_top[1], self.right_sleeve_bottom[1], 50)
        thresh_line = get_area_by_pct(self.right_sleeve_top[1], self.right_sleeve_bottom[1], 10)
        
        
        crop = dilation(canny_edge(mask[self.right_sleeve_top[1]:self.right_sleeve_top[1]+thresh_bottom, self.right_width_top[0]-thresh_right:self.right_width_top[0]]))
        crop_approx = detect_approx_contours(crop, threshold=0.01)
        width_right_coords = [(c[0][0], c[0][1]) for c in crop_approx]
        _, max_coord = split_by_x(width_right_coords)

        
        right_x = (x1-(x1-(self.right_width_top[0]-thresh_right)))+max_coord[0]
        right_y = (y1-(y1-self.right_sleeve_top[1]))+max_coord[1]
        self.right_sleeve_width = (right_x, right_y)
        draw_circle(image, self.right_sleeve_width , color=(0, 0, 255))
        
        crop = dilation(mask[:self.right_sleeve_width[1], self.right_sleeve_width[0]:self.right_sleeve_width[0]+thresh_line])
        crop_approx = detect_approx_contours(crop, threshold=0.01)
        width_right_coords = [(c[0][0], c[0][1]) for c in crop_approx]
        min_coord, _ = split_by_y(width_right_coords)
        
        right_x = (x1-(x1-(self.right_sleeve_width[0]-thresh_line)))+min_coord[0]
        self.right_sleeve_width = (right_x, min_coord[1])
        
        draw_circle(image, self.right_sleeve_width , color=(0, 255, 0))
        
        length = np.linalg.norm(np.array([self.right_sleeve_width]) - np.array([self.right_sleeve_top]))
        length = (length * self.coin_per_mm) * inche_per_mm
        cv2.putText(image, str(f"Sleeve right Width : {length:.2f} Inch"), (w-800, 280), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 255), 5, cv2.LINE_AA)
        
        draw_two_sided_arrow(image, self.right_sleeve_width, self.right_sleeve_top, (255, 0, 255), 6, 0.05)
        
    
    def detection(self, image):
        orignal_image = image.copy()
        (min_contours, min_bbox), (max_contours, max_bbox), max_approx = detect_min_max_contours(image, threshold=0.01)
        [draw_circle(image, tuple(apr[0]), color=(0, 255, 0)) for apr in max_approx] 

        cv2.rectangle(image, min_bbox[0], min_bbox[2], (0, 0, 255), 2)
        cv2.rectangle(image, max_bbox[0], max_bbox[2], (0, 0, 255), 2)

        try:
            self.pixels_in_mm = self.coin_length_pixels(image, min_bbox)
        except:
            pass

        try:
            self.sleeve_length(image, max_contours, max_bbox, max_approx)
        except:
            pass

        try:
            self.width_top(image, max_contours, max_bbox, max_approx)
        except:
            pass

        try:
            self.width_mid(image, max_contours, max_bbox, max_approx)
        except:
            pass

        try:
            self.length_right(orignal_image, image, max_contours, max_bbox, max_approx)
        except:
            pass
        
        try:
            self.sleeve_width_left(orignal_image, image, max_contours, max_bbox, max_approx)
        except:
            pass
        
        # try:
        self.sleeve_width_right(orignal_image, image, max_contours, max_bbox, max_approx)
        # except:
        #     pass
        
        show(image)
        cv2.imwrite('images/output.jpg', image)