import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from skimage.feature import hog
from scipy.ndimage.measurements import label

class Vehicle():
    def __init__(self):
        # was the vehicle detected in the last iteration?
        self.detected = False  
        self.n_detected = 0
        self.n_nondetected = 0
        self.xpixels= None
        self.ypixels= None
        

        self.recent_xfitted = [] 
        self.bestx = None     

        self.recent_yfitted = [] 
        self.besty = None     

        self.recent_wfitted = [] 
        self.bestw = None  

        self.recent_hfitted = [] 
        self.besth = None  

def process_video(in_file):
    count = 0
    video_cap = cv2.VideoCapture(in_file)
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    is_writer_inited = False
    while(video_cap.isOpened()):
        # read a image from video file
        ret, frame = video_cap.read()
        if not is_writer_inited:
            out_video = cv2.VideoWriter(self.out_path + 'output.avi',fourcc, 30.0, frame.shape[:2][::-1])
            is_writer_inited = True
        if frame is None:
            print('no more video frame')
            break
        out_frame = self._process_image(frame, True)
        out_video.write(out_frame)

