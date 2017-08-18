import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import time
import cv2
import os
from moviepy.editor import *

from help_functions import *

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from skimage.feature import hog
from scipy.ndimage.measurements import label

def apply_threshold(heat_map, threshold):
    #zero out pixels below the threshold
    heat_map[heat_map < threshold] = 0
    return heat_map
# find the bboxes based on labels
def calc_labeld_bboxes(labels):
    bboxs = []
    for car_number in range(1, labels[1]+1):
        # Find the pixels with each 
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxs.append(bbox)
    return bboxs
# draw rectangle boxes based on coorinates from bboxes  
def draw_bboxes(img, bboxs):
    for bbox in bboxs:
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img        = np.copy(img)
    draw_img        = np.copy(img)
    
    img_tosearch    = img[ystart:ystop,:,:]
    ctrans_tosearch = img_tosearch

    # convert to YCrCb as the mode was trianed based on YCrCb images 
    ctrans_tosearch = convert_color(ctrans_tosearch, conv='RGB2YCrCb')
    # scale to [0, 1] as the model was tranied based on png images
    ctrans_tosearch = ctrans_tosearch.astype(np.float32) / 255

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    heat_map = np.zeros_like(img[:,:,0])
    heat_map = heat_map.astype(np.uint8) 

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks        = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks        = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window             = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step     = 2  # Instead of overlap, define how many cells to step
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # print('number of windows ' + str(nxsteps*nysteps))

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1    = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2    = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3    = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop  = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features    = color_hist(subimg, nbins=hist_bins)

            test_features    = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features    = X_scaler.transform(test_features) 
            test_prediction  = svc.predict(test_features)
            

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            if test_prediction == 1:
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),10) 
                heat_map[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
    return draw_img, heat_map

dist_pickle    = pickle.load(open("svc_pickle_svc_linear.p", "rb" ) )
svc            = dist_pickle["svc"]
X_scaler       = dist_pickle["scaler"]
orient         = dist_pickle["orient"]
pix_per_cell   = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size   = dist_pickle["spatial_size"]
hist_bins      = dist_pickle["hist_bins"]
color_space    = dist_pickle["color_space"]
spatial_feat   = dist_pickle["spatial_feat"] # Spatial features on or off
hist_feat      = dist_pickle["hist_feat"]    # Histogram features on or off
hog_feat       = dist_pickle["hog_feat"]     # HOG features on or off
hog_channel    = dist_pickle["hog_channel"]  # Can be 0, 1, 2, or "ALL"

def output_hog_features(img_file, orient, pix_per_cell, cell_per_block):
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    return hog_image

def bb_inter_over_union_ratio(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    if xB < xA or yB < yA:
        return 0.0
    else:
        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
        # calculate the are of each rectangle
        boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
        boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
        # calculate the ration between area of intersection 
        # and the minimum area of two rectangle
        iou = interArea / min(float(boxAArea), float (boxBArea))
        # iou = max(0.0, iou)
        return max(0.0, iou)


class Vehicle():
    def __init__(self, bbox):
        # was the vehicle detected in the last iteration?
        self.n_detected    = 1
        self.n_nondetected = 0
        self.bbox      = bbox
        self.vechile_w = bbox[1][0] - bbox[0][0]
        self.vechile_h = bbox[1][1] - bbox[0][1]
        self.xpixels   = (bbox[0][0], bbox[1][0])
        self.ypixels   = (bbox[0][1], bbox[1][1])        

    # we only consider a car in good size if 
    # both width and height is larger than 10 pixels 
    def in_good_size(self, size_threshold = 10):
        return  min(self.vechile_w, self.vechile_h) > size_threshold

    # if a previously detected vehicle is not detected in current frame
    # we shrink the size of this vechile a bit to reflect that we are less
    # confident about this vehicle, if we keep not detecting this vehicle
    # the size will keep shrinking until it too small and we will just remove it
    def shrink_current_vehicle(self, shrink_ratio = 0.2):
        bbox = self.bbox
        x0 = bbox[0][0] + int(self.vechile_w * shrink_ratio)
        y0 = bbox[0][1] + int(self.vechile_h * shrink_ratio)

        x1 = x0 + int(self.vechile_w * (1-shrink_ratio)) 
        y1 = y0 + int(self.vechile_h * (1-shrink_ratio))
        
        bbox = ((x0,y0), (x1,y1))
        self.vechile_w = x1 - x0
        self.vechile_h = y1 - y0
        self.xpixels   = (x0, x1)
        self.ypixels   = (y0, y1)
        self.bbox = bbox
        self.n_detected    = self.n_detected - 1
        self.n_detected    = max(self.n_detected, 0)
        self.n_nondetected = self.n_nondetected + 1
        self.n_nondetected = min(self.n_nondetected, 10)
    # we detected an existing vehicle once again with a slightly 
    # different location. 
    # use the newly detected location to refresh this vechile 
    def refresh_current_vehicle(self, bbox):
        self.n_detected += 1
        self.n_nondetected -= 1

        self.n_detected = min(self.n_detected, 10)  
        self.n_nondetected = max(0, self.n_nondetected)  

        new_xpixels = np.array((bbox[0][0], bbox[1][0]), dtype = 'f')
        new_ypixels = np.array((bbox[0][1], bbox[1][1]), dtype = 'f')
        old_xpixels = np.array(self.xpixels, dtype = 'f')
        old_ypixels = np.array(self.ypixels, dtype = 'f')
        # bias towards the location from history detection
        avg_x = new_xpixels * 0.25 + old_xpixels * 0.75
        avg_y = new_ypixels * 0.25 + old_ypixels * 0.75

        self.xpixels = avg_x.astype(np.int32).tolist()
        self.ypixels = avg_y.astype(np.int32).tolist()
        self.bbox    = ((self.xpixels[0], self.ypixels[0]), (self.xpixels[1], self.ypixels[1]))
        self.vechile_w = self.bbox[1][0] - self.bbox[0][0]
        self.vechile_h = self.bbox[1][1] - self.bbox[0][1]

# object to hold raw detection results
class FrameDetectionResult():
    def __init__(self, image, bboxes, name_hint=None):
        # was the vehicle detected in the last iteration?
        self.image         = image  
        self.name_hint     = name_hint
        self.bboxes        = bboxes # raw detection result 
        self.n_bboxes      = len(bboxes)
        self.vechiles_list = None  # vechiles_list based on raw bboxes detection results

# compare to bbox to decide whether they are describing the same car
def compare_bbox(bbox1, bbox2, iou_threshold = 0.3):
    if bbox1 == None or bbox2 == None:
        return False
    iter_union_ratio = bb_inter_over_union_ratio(bbox1, bbox2)
    return True if iter_union_ratio > iou_threshold else False


# object to perform actial vehicle detection
class VehiclesDetection():
    def __init__(self, out_path = 'tmp/', is_debug_mode = False, is_image_mode = False, decisions_window_size = 10):
        self.out_path     = out_path
        self.dbg_out_path = None
        self.is_image_mode = is_image_mode
        self.is_debug_mode = is_debug_mode
        self.n_frames = 0 # how many frame has been processed 
        # create folders if they does not exit 
        if out_path and not os.path.exists(out_path):
            os.makedirs(out_path)
        # create folder to output some intermedian image results 
        if self.is_debug_mode:
            self.dbg_out_path = os.path.join(self.out_path, 'debug_out')
            if self.dbg_out_path and not os.path.exists(self.dbg_out_path):
                os.makedirs(self.dbg_out_path)

        self.decisions_window_size = decisions_window_size
        self.decisions_window      = [None] * decisions_window_size
        self.detection_thresold    = 3
        self.vechiles_list         = []

    def find_best_match(self, bboxes):
        n_vehicles = len(self.vechiles_list)
        n_bboxes = len(bboxes)
        scores = np.zeros(n_vehicles*n_bboxes).reshape(n_vehicles, n_bboxes)

        for i in range(n_vehicles):
            vechile = self.vechiles_list[i]
            # check with existing vechile is in newly detecting list
            for j in range(n_bboxes): 
                bbox = bboxes[j]
                score = bb_inter_over_union_ratio(vechile.bbox, bbox)
                scores[i][j] = score
        vechiles_list_tmp = list(self.vechiles_list)
        bboxes_tmp = list(bboxes)

        new_vechiles_list = []
        while scores.size > 0 and np.max(scores) > 0.3:
            i,j = np.unravel_index(scores.argmax(), scores.shape)
            # find current best score
            if scores[i,j] > 0.3:
                vechile = vechiles_list_tmp[i]
                vechile.refresh_current_vehicle(bboxes_tmp[j])
                new_vechiles_list.append(vechile)

                del vechiles_list_tmp[i]
                del bboxes_tmp[j]

                scores = np.delete(scores, i, axis=0)
                if scores.size > 0:
                    scores = np.delete(scores, j, axis=1)

        for v in vechiles_list_tmp:
            v.n_detected =  v.n_detected - 1
            v.n_nondetected = v.n_nondetected + 1
            v.n_detected = max(v.n_detected, 0)  
            v.n_nondetected = min(10, v.n_nondetected)  

            old_bbox = v.bbox
            oldw = v.vechile_w
            oldh = v.vechile_h                
            v.shrink_current_vehicle()
            print('no vechile detected for this frame, shrinking', old_bbox, 'to', vechile.bbox)
            if v.n_nondetected >= v.n_detected or not v.in_good_size():
                print('removing vechile: good size?', v.in_good_size(), v.bbox, v.n_nondetected, v.n_detected)
                vechiles_list_tmp.remove(v)

        self.vechiles_list = vechiles_list_tmp
        if len(bboxes_tmp) > 0:
            for bbox in bboxes_tmp:
                new_vechile = self.create_new_vechile(bbox)
                new_vechiles_list.append(new_vechile)
        self.vechiles_list += new_vechiles_list

    def draw_vechiles(self, image):
        for vechile in self.vechiles_list:
            cv2.rectangle(image, vechile.bbox[0], vechile.bbox[1], (0, 0, 255), 6)
        return image
    # only draw the vehcile on image if we have detected it for 
    # more than a few times 
    def draw_vechiles_with_confidence(self, image):
        good_vechile = []
        for vechile in self.vechiles_list:
            if vechile.n_detected >= self.detection_thresold:
                cv2.rectangle(image, vechile.bbox[0], vechile.bbox[1], (0, 0, 255), 6)
                # print('***drawing n_detection', vechile.n_detected, 
                #     'n_notdetection', vechile.n_nondetected, 'pos', vechile.bbox, 'size',(vechile.vechile_w, vechile.vechile_h))
        return image

    # update current vechile list with bboxes from current image
    def update_vechiles_list(self, bboxes):
        if bboxes == None or len(bboxes) == 0: # non detected vechiles in current frame
            for vechile in self.vechiles_list:
                old_bbox = vechile.bbox
                oldw = vechile.vechile_w
                oldh = vechile.vechile_h                
                vechile.shrink_current_vehicle()
                print('no vechile detected for this frame, shrinking', old_bbox, 'to', vechile.bbox)
                if vechile.n_nondetected >= vechile.n_detected or not vechile.in_good_size():
                     print('removing vechile: good size?', vechile.in_good_size(), vechile.bbox, vechile.n_nondetected, vechile.n_detected)
                     self.vechiles_list.remove(vechile)
        else:
            self.find_best_match(bboxes)
                    
        return self.vechiles_list

    def create_new_vechile(self, bbox):
        print('adding a new vechile', bbox)
        vehicle = Vehicle(bbox)
        for i in range(1, self.decisions_window_size):
            past_result = self.decisions_window[i]
            if past_result == None or past_result.vechiles_list == None:
                continue
            for p_vehicle in past_result.vechiles_list:
                if compare_bbox(p_vehicle.bbox, bbox):
                    print('found a history results on frame:', i, "position:", p_vehicle.bbox)
                    vehicle.n_detected = min(p_vehicle.n_detected, 3)
                    break
        return Vehicle(bbox)
    # vechiles detected on current input image
    def add_to_decision_window(self, fdr):
        decison_window = self.decisions_window
        window_size = self.decisions_window_size
        current_idx = 0
        if fdr is not None:
            self.n_frames += 1
            print('adding detection results from', fdr.name_hint)
        # shift the decision window list to right by one
        for idx in range(window_size -1 , 0, -1):
            decison_window[idx] = decison_window[idx-1]
            decison_window[0] = fdr

        result = decison_window[current_idx] 
        name = result.name_hint if result and result.name_hint else ('out_%s.jpg' % (self.n_frames))

        fdr.vechiles_list = self.update_vechiles_list(result.bboxes)
        img = self.draw_vechiles_with_confidence(result.image)
        if self.is_debug_mode:
            print('writing image...', name)          
            cv2.imwrite(self.out_path  + name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img


    def process_image(self, image, imgfile_name = None):        
        x_start_stop = [0, image.shape[1]]
        y_start_stop = [image.shape[0]//2, image.shape[0]]
        ystart = y_start_stop[0]
        ystop  = y_start_stop[1]

        scales = [1.15, 1.25, 1.75, 2, 2.25, 2.5, 3]
        heat_map = np.zeros_like(image[:,:,0])
        heat_map = heat_map.astype(np.uint8) 
        t1=time.time()
        for scale in scales:
            out_img, current_heat_map = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            heat_map += current_heat_map
            # if self.is_debug_mode:
            #     scale_name = 'scale_' + str(scale)
            #     scale_folder = os.path.join(self.dbg_out_path, scale_name)
            #     if not os.path.exists(scale_folder):
            #         os.makedirs(scale_folder)
            #     img_file     = os.path.join(scale_folder, imgfile_name)
            #     heatmap_file = os.path.join(scale_folder, 'heatmap_' + imgfile_name)
            #     cv2.imwrite(img_file, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            #     current_heat_map = np.clip(current_heat_map * 255, 0, 255)
            #     cv2.imwrite(heatmap_file, current_heat_map)

        labels = label(heat_map)
        car_bboxs = calc_labeld_bboxes(labels)
        t2=time.time()
        if not self.is_image_mode:
            fdr = FrameDetectionResult(image, car_bboxs, imgfile_name)
            return self.add_to_decision_window(fdr)
        else:
            out_img = draw_bboxes(image, car_bboxs)
            imgfile = os.path.join(self.out_path, imgfile_name)
            cv2.imwrite(imgfile, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            heatmap_file = os.path.join(self.out_path, 'heatmap_' + imgfile_name)
            heat_map = np.clip(heat_map * 255, 0, 255)
            cv2.imwrite(heatmap_file, heat_map)


test_images = 'test_images/'
test_images_out = 'output_images/'
img_files   = sorted(glob.glob(test_images + '*jpg'))

def generate_Writeup_results(img_files, out_path):
    detector = VehiclesDetection(out_path, is_debug_mode=True, is_image_mode=True)
    hog_out_path = os.path.join(out_path, 'hog_out')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(hog_out_path):
        os.makedirs(hog_out_path)

    for img in img_files:
        img_name = img.split('/')[-1]
        hog_img_file = os.path.join(hog_out_path, img_name)
        hog_img = output_hog_features(img, orient, pix_per_cell, cell_per_block)
        #hog_img is in [0,1] float image, need convert to [0, 255] before writing to disk
        hog_img = np.clip(hog_img * 255, 0, 255)
        cv2.imwrite(hog_img_file, hog_img)

        image = mpimg.imread(img)
        detector.process_image(image, img_name)

# generate_Writeup_results(img_files, test_images_out)

# test_images = 'test_video_small/'
# test_images_out = 'tmp2_out/'
# img_files   = sorted(glob.glob(test_images + '*jpg'), key=os.path.getmtime)
# vdetector = VehiclesDetection(test_images_out, is_debug_mode=True, is_image_mode=False)

# for img_file in img_files:
#     image = mpimg.imread(img_file)
#     name  = img_file.split('/')[-1]
#     vdetector.process_image(image, name)

in_file = 'test_video.mp4'
vd  = VehiclesDetection()

clip = VideoFileClip(in_file)
white_clip = clip.fl_image(vd.process_image) #NOTE: this function expects color images!! 
white_clip.write_videofile('test_video_out.mp4', audio=False)
HTML("""
<video width="1280" height="720" controls>
<source src="{0}">
</video>
""".format(processed_file))
