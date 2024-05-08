import cv2
import numpy as np


def get_intersection(x_min, y_min, x_max, y_max, contour, threshold=30):
    if x_min == x_max:
        x = int(x_min)
        for point in contour:
            if abs(point[0][0] - x) <= threshold:
                if y_min <= point[0][1] <= y_max:
                    return point[0][0], point[0][1]
        return None
    else:
        m = (y_max - y_min) / (x_max - x_min)
        b = y_min - m * x_min
        for i in range(len(contour) - 1):
            point1 = contour[i][0]
            point2 = contour[i+1][0]

            x1, y1 = point1
            x2, y2 = point2

            if (x_min <= x1 <= x_max or x_min <= x2 <= x_max) and (y_min <= y1 <= y_max or y_min <= y2 <= y_max):
                if x1 == x2:
                    if abs(x1 - x_min) <= threshold:
                        y = m * x1 + b
                        if y1 <= y <= y2 or y2 <= y <= y1:
                            return int(x1), int(y)
                else:
                    k = (y2 - y1) / (x2 - x1)
                    n = y1 - k * x1
                    if abs(m - k) <= threshold:
                        if m>0 or k>0:
                            x = (n - b) / (m - k)
                            if x1 <= x <= x2 or x2 <= x <= x1:
                                return int(x), int(m * x + b)
        return None


def min_idx(lst):
    arr = np.array(lst)
    return np.lexsort((arr[:, 1], -arr[:, 0]))[0]


def max_idx(lst):
    arr = np.array(lst)
    return np.lexsort((-arr[:, 1], arr[:, 0]))[0]


def get_nearest_values_with_indices(data, target, threshold=2):
    arr = np.array(data)
    idx = np.abs(arr - target) <= threshold
    return [(i, v) for i, v in np.ndenumerate(arr[idx])]


def get_area_by_pct(num1, num2, pct):
    return int(np.abs(num1 - num2) * pct / 100)


def split_by_y(lst):
    # convert the list of tuples to a NumPy array
    arr = np.array(lst)

    # get the unique y values
    y_values = np.unique(arr[:, 1])

    # create two new arrays based on y values
    arr_min = arr[arr[:, 1] == y_values[0]]
    arr_max = arr[arr[:, 1] == y_values[-1]]

    # convert the new arrays to lists
    tpl_min = tuple(arr_min.tolist()[0])
    tpl_max = tuple(arr_max.tolist()[0])

    return tpl_min, tpl_max


def split_by_x(lst):
    # convert the list of tuples to a NumPy array
    arr = np.array(lst)

    # get the unique x values
    x_values = np.unique(arr[:, 0])

    # create two new arrays based on x values
    arr_min = arr[arr[:, 0] == x_values[0]]
    arr_max = arr[arr[:, 0] == x_values[-1]]

    # convert the new arrays to lists
    tpl_min = tuple(arr_min.tolist()[0])
    tpl_max = tuple(arr_max.tolist()[0])

    return tpl_min, tpl_max


def center_value(num1, num2):
    arr = np.array([num1, num2])
    return int(np.median(arr))


def get_center_point_by_x(lst):
    arr = np.array(lst)
    x_sorted = np.sort(arr[:, 0])
    midpoint = (x_sorted[0] + x_sorted[-1]) / 2
    closest_x_indices = np.argsort(np.abs(x_sorted - midpoint))[:2]
    closest_tuples = arr[closest_x_indices]
    x_avg = np.mean(closest_tuples[:, 0])
    y_avg = np.mean(closest_tuples[:, 1])
    return (int(x_avg), int(y_avg))


def get_bounding_box(contours):
    x, y, w, h = cv2.boundingRect(contours)
    return (x, y), (x+w, y), (x+w, y+h), (x, y+h)


def show(image):
    image = cv2.resize(image, (795, 640))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()