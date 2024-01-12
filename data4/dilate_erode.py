

import cv2
import numpy as np

def dilate_img(img,a,iterations):
    kernel = np.ones((a,a),np.uint8)
    dilate = cv2.dilate(img,kernel,iterations = iterations)
    return dilate
def erosion_img(img,a,iterations):
    kernel = np.ones((a,a),np.uint8)
    erosion = cv2.erode(img, kernel, iterations= iterations)
    return erosion

def nothing(pos):
    pass

with open('./data/DUTS/train.txt', 'r') as lines:
    samples = []
    for line in lines:
        maskpath  = './data/DUTS/mask/'  + line.strip() + '.png'   
        img = cv2.imread(maskpath,1)
        dilate = dilate_img(img,3,5)
        erode = erosion_img(img, 3,5) 
        cv2.imwrite('./data/DUTS/mask_d3/'+ line.strip() + '.png',dilate)
        cv2.imwrite('./data/DUTS/mask_e3/'+ line.strip() + '.png',dilate)
