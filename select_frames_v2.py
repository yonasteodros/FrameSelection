#!/usr/bin/python2

import sys
from glob import glob
from os import path
import cv2 as cv
import shutil

import numpy as np


# Settings
l_buff = 2000
#t_sharp = 3
t_sharp = 1.3
t_distance = 70
no_min = 0.05 / 0.95
no_max = 0.20 / 0.80
min_sep = 10  # Minimum delay before new frame
max_sep = 30 # Maximum delay before reset

# Gaussian filter std=1
H_g = np.array([
    [0.0113,   0.0838,   0.0113],
    [0.0838,   0.6193,   0.0838],
    [0.0113,   0.0838,   0.0113]])

# Laplacian filter
H_l = np.array([
    [0.1667,   0.6667,   0.1667],
    [0.6667,  -3.3333,   0.6667],
    [0.1667,   0.6667,   0.1667]])

# KP detector / matcher
detector = cv.ORB(1000)
matcher = cv.BFMatcher(cv.NORM_HAMMING)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_yuv(img_path):
    I_bgr = cv.imread(img_path, cv.IMREAD_COLOR)
    I_yuv = cv.cvtColor(I_bgr, cv.COLOR_BGR2YUV)
    return np.array(I_yuv[:,:,0])

def get_img_list(img_dir):
    imgs = glob(path.join(img_dir, '*.png'))
    imgs.sort()
    return imgs

"""
def sharpness(I):
    return cv.Laplacian(I, cv.CV_64F).var()
"""

def sharpness(I):
    I_smooth = cv.filter2D(I, cv.CV_64FC1, H_g)
    I_l = cv.filter2D(I_smooth, cv.CV_64FC1, H_l)
    I_lg = cv.filter2D(I_l, cv.CV_64FC1, H_g)    
    return np.std(I_l - I_lg)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++




#+++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    img_dir = '/home/yo/Documents/Mike/frames/'
    out_dir = '/home/yo/Documents/Mike/images/'

    img_list = get_img_list(img_dir)

    selected = []
    des_buff = None
    
    i = -1
    i_last = 0 # last selected frame

    for img in img_list:

        i += 1

        I = load_yuv(img)

        I = cv.equalizeHist(I)

        # computes sharpness
        S = sharpness(I)

        print S
        if S < t_sharp:
            continue

        # des_buff == None --> used for the initialisation / when the buffer is empty
        # i - i_last > max_sep -->
        if des_buff == None or i - i_last > max_sep:
            _, des_buff = detector.detectAndCompute(I, None)
            
            selected.append(img)
            i_last = i
            continue
        
        _, des = detector.detectAndCompute(I, None)

        match = matcher.match(des, des_buff) #  It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.

        # good_match = [m for m in match if m.distance < t_distance]

        bool_match = [m.distance < t_distance for m in match]
        
        # n_old = len(good_match)

        n_old = sum(bool_match)
        n_new = des.shape[0] - n_old
        print img, n_old, n_new
        
        no = 1.0 * n_new / n_old
        
        if no > no_min and no < no_max and i - i_last > min_sep:
            # Good frame
            selected.append(img)
            i_last = i
            
            # new_idx = np.array([m.distance >= t_distance for m in match]) # array of booleans

            new_idx = np.array([not x for x in bool_match])  # array of booleans

            # bool_idx = sum([a==b for a,b in zip(new_idx,new_idx2)])

            new_des = des[new_idx, :] # select only those values with index True
            
            des_buff = np.concatenate((new_des, des_buff)) # concatenate the new keypoints with the old ones

            des_buff = des_buff[0:l_buff,:]
    
    for i in range(0, len(selected)):
        fname = path.split(selected[i])[1]

        shutil.copyfile(path.join(img_dir, fname), path.join(out_dir, "3%04d.png" % i))

        # img = cv.imread(path.join(img_dir, fname))
        # cv.imwrite(path.join(out_dir, "%04d.png" % i), img, [int(cv.cv.CV_IMWRITE_PNG_COMPRESSION), 1])
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
