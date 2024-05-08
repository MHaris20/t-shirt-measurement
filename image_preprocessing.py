import cv2
import numpy as np

def mask_method_1(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    sigma = 20.0
    theta = 3.1
    lambd = 20
    gamma = 1.0
    psi = 0.0

    mean_block = 5
    mean_c = 1

    g_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(gray_image, cv2.CV_8UC1, g_kernel)

    filtered_img = cv2.adaptiveThreshold(filtered_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, mean_block, mean_c)
    filtered_img = cv2.Canny(filtered_img, 50, 100)

    # filtered_img = cv2.dilate(filtered_img, None, iterations=1)
    filtered_img = cv2.threshold(filtered_img, 120, 255, cv2.THRESH_BINARY_INV)[1]

    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
    return rgb_img


def mask_method_2(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    rgb_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return rgb_img


def mask_method_3(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kernel_size = 3
    sigma = 0.1
    theta = 0.0
    lambd = 1
    gamma = 1.0
    psi = 0.0

    mean_block = 5
    mean_c = 8
    
    erode = 2
    
    filtered_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, mean_block, mean_c)
    
    g_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(filtered_img, cv2.CV_8UC1, g_kernel)
    
    kernel = np.ones((3, 3), np.uint8)
    filtered_img = cv2.erode(filtered_img, kernel=kernel, iterations=erode)
    
    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
    return rgb_img


def erotion(frame, k=3):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((k, k), np.uint8)
    filtered_img = cv2.erode(gray_image, kernel=kernel, iterations=1)
    
    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
    return rgb_img


def dilation(frame, k=3):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((k, k), np.uint8)
    filtered_img = cv2.dilate(gray_image, kernel=kernel, iterations=1)
    
    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
    return rgb_img


def canny_edge(frame, lower=50, upper=100):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    filtered_img = cv2.Canny(gray_image, lower, upper)
    
    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
    return rgb_img
