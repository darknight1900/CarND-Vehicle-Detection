import numpy as np
import cv2
import glob
import time
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle

from skimage.feature import hog
from help_functions import *

CSV_LABEL = 'training_data/udacity_labeled_dataset/object-dataset/labels.csv'

def train_car_classifier(cars, notcars, color_space='YUV', 
    orient=9, pix_per_cell=8, cell_per_block=2, 
    hog_channel='ALL', spatial_size=(32, 32), hist_bins=32, 
    spatial_feat=True, hist_feat=True, hog_feat=True, save_model=False):

                            
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    print('car_features ' + str(len(car_features)) + ' notcar_features ' + str(len(notcar_features)))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.3, random_state=rand_state)

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    t=time.time()
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # save the training results for later usage 
    dist_pickle = {}
    dist_pickle["svc"]            = svc
    dist_pickle["scaler"]         = X_scaler
    dist_pickle["orient"]         = orient
    dist_pickle["pix_per_cell"]   = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"]   = spatial_size
    dist_pickle["hist_bins"]      = hist_bins
    dist_pickle["color_space"]    = color_space
    dist_pickle["hog_channel"]    = hog_channel
    dist_pickle["spatial_feat"]   = spatial_feat
    dist_pickle["hist_feat"]      = hist_feat
    dist_pickle["hog_feat"]       = hog_feat
    if save_model:
        pickle.dump(dist_pickle, open("svc_pickle_svc_linear.p", "wb" ), 0)
    
        

car_path  = 'training_data/large/vehicles/'
noncar_path    = 'training_data/large/non-vehicles/'
cars           = [filename for filename in glob.iglob(car_path + '**/*.png', recursive=True)]
notcars        = [filename for filename in glob.iglob(noncar_path + '**/*.png', recursive=True)]

cars = shuffle(cars)
notcars = shuffle(notcars)

num_of_cars    = len(cars);
num_of_notcars = len(notcars);

print('car samples ' + str(num_of_cars) + ' notcar samples ' + str(num_of_notcars))

sample_size   = min(num_of_cars, num_of_notcars)
sample_size   = 6000
cars          = cars[0:sample_size]
notcars       = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space    = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9  # HOG orientations
pix_per_cell   = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel    = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size   = (32, 32) # Spatial binning dimensions
hist_bins      = 32    # Number of histogram bins
spatial_feat   = True # Spatial features on or off
hist_feat      = True # Histogram features on or off
hog_feat       = True # HOG features on or off

train_car_classifier(cars=cars, notcars=notcars, color_space=color_space, save_model=True)


