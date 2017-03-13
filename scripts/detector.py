#work around for training from parent folder, else features script can't be found
import sys
import os
sys.path.append( os.path.dirname(os.path.realpath(__file__)))

#___________________Code_______________________________________
from sklearn import svm
from sklearn.externals import joblib
from features import FeatureExtractor
#from scipy.ndimage import imread
import random
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle

#work around because of strange behaviour of StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt

class CarDetector:

    # Methods

    def __init__(self):
        self.clf = svm.SVC(kernel = "rbf")
        self.heatmaps = []
        self.frame_counter = 0
        for i in range(3):
            self.heatmaps.append(np.ones((720,1280)))


    def train(self, train_data, train_labels):
        self.clf.fit(train_data, train_labels)
        return True


    def classify(self, image_list):
        """
        Classify list of images.
        """
        features = [FeatureExtractor.get_image_features(image) for image in image_list]
        normalized_features = [FeatureExtractor.normalize_features(feature) for feature in features ]

        #classification = [self.clf.predict(x) for x in normalized_features]
        classification = self.clf.predict(normalized_features)
        return classification


    def is_car(self, image):
        """
        Classify single image.
        """
        features = FeatureExtractor.get_image_features(image)
        normalized_features = FeatureExtractor.normalize_features(features)
        image_is_car = self.clf.predict(normalized_features)
        if (image_is_car):
            return True
        return False


    def accuracy(self, test_data, test_labels):
        return self.clf.score(test_data, test_labels)


    def save(self, filename="svc.model"):
        joblib.dump(self.clf, filename) 


    def load(self, filename="svc.model"):
        self.clf = joblib.load(filename) 


    def detect(self, image, object_size=(32,32), stride=16):
        """
        Detect all objects in a image.
        """
        Y,X,Z = image.shape
        horizontal_steps = int(X / object_size[0])
        vertical_steps = int(Y / object_size[1])

        rect_points=[]
        image_parts=[]
        subframe_positions=[]
        rect_points = []
        global_y_start = int(Y/1.7)
        magicvalue = 40 # own car at bottom of the image
        global_y_end = Y-object_size[1]-magicvalue

        for i in range(0,X-object_size[0], stride):
            for j in range(global_y_start , global_y_end, stride):
                x_start = i
                y_start = j
                x_end = i+object_size[0]
                y_end = j+object_size[1]
                img_part = image[y_start:y_end, x_start:x_end, :]
                image_parts.append(img_part)
                subframe_positions.append([x_start, y_start, x_end, y_end])
        results = self.classify(image_parts)
        for i in range(len(results)):
            if (results[i] == True):
                rect_points.append(subframe_positions[i])

        return rect_points


    def detect_multiscale(self, image, scale=1.05, min_sliding_window=(32,32), max_sliding_window=(256,256)):
        all_window_rect_points = []
        while ((min_sliding_window[0] <= max_sliding_window[0]) & (min_sliding_window[1] <= max_sliding_window[1])):
            stride = (int)(min_sliding_window[0]/2)
            bBoxes = self.detect(image, object_size=min_sliding_window, stride=stride)
            all_window_rect_points.append(bBoxes)
            min_sliding_window = (int(min_sliding_window[0] * scale), int(min_sliding_window[1] * scale))
        return all_window_rect_points


    def detect_full_pipeline(self, rgb_image, min_sliding_window=(64,64), max_sliding_window=(128,128)):
        bBoxes = self.detect_multiscale(rgb_image, scale=1.5, min_sliding_window=min_sliding_window, max_sliding_window=max_sliding_window)

        heatmap = CarDetector.get_heatmap(bBoxes, rgb_image.shape[0:2])
        heatmap_contours = CarDetector.get_countours_of_heatmap(heatmap)
        clean_heatmap = CarDetector.clean_heatmap(rgb_image, heatmap_contours, heatmap)

        self.heatmaps.append(clean_heatmap)
        self.heatmaps.pop(0)

        combined_heatmap_binary = np.all(self.heatmaps, axis=0)
        combined_heatmap = np.zeros_like(clean_heatmap)
        combined_heatmap[combined_heatmap_binary == True] = clean_heatmap[combined_heatmap_binary == True]

        from test import display_heatmap
        from test import display_image
        from test import compare_before_after 
        #display_heatmap(heatmap)
        #display_heatmap(clean_heatmap)
        #display_heatmap(combined_heatmap)

 
        contours = CarDetector.get_countours_of_heatmap(combined_heatmap)      
        output = CarDetector.heatmap_contours_to_bBoxes(rgb_image, contours, combined_heatmap)
        self.frame_counter = self.frame_counter + 1
        #display_image(output)
        #compare_before_after(combined_heatmap, output, "Heatmap", "Output image")



        return output

    # END Methods

    # Static Methods

    @staticmethod
    def clean_heatmap(rgb_image, heatmap_contours, heatmap, threshold=2):
        for cnt in heatmap_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            heatmap_crop = heatmap[y:y+h,x:x+w]
            local_max_index = np.argmax(heatmap_crop)
            center_index = np.unravel_index(local_max_index, (h,w))
            local_max = heatmap_crop[center_index[0], center_index[1]]

            aspectRatioCheck = (w < 2 * h) & (h < 1.5 * w)
            ret, thresh = cv2.threshold(heatmap_crop.astype(np.uint8), 0, 1,cv2.THRESH_BINARY)
            bBoxIsFilledCheck = (np.sum(thresh) >= 0.7*w*h)

            if not((local_max > threshold) & bBoxIsFilledCheck & aspectRatioCheck):
                heatmap[y:y+h,x:x+w]=0

        return heatmap

    @staticmethod
    def get_heatmap(bBoxes, shape):

        heatmap = np.zeros(shape).astype(int)
        for scaled_bBoxes in bBoxes:
            for bBox in scaled_bBoxes:
                x_start = bBox[0]
                y_start = bBox[1]
                x_end = bBox[2]
                y_end = bBox[3]
                heatmap[y_start:y_end, x_start:x_end] = heatmap[y_start:y_end, x_start:x_end] + 1
        return heatmap


    @staticmethod
    def get_countours_of_heatmap(heatmap):
        heatmap_u8c1 = heatmap.astype(np.uint8)
        ret, thresh = cv2.threshold(heatmap_u8c1, 0,1,cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("Cnts: ", len(contours_hierarchy))
        return contours


    @staticmethod
    def heatmap_contours_to_bBoxes(image, contours, heatmap, threshold=2):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            heatmap_crop = heatmap[y:y+h,x:x+w]
            local_max_index = np.argmax(heatmap_crop)
            center_index = np.unravel_index(local_max_index, (h,w))
            local_max = heatmap_crop[center_index[0], center_index[1]]
            y_for_bBox = center_index[0] + y - 20
            x_for_bBox = center_index[1] + x - 55

            check_val = w*h

            bBoxSizeCheck = heatmap.shape[0] * heatmap.shape[1]
            bBoxBigEnough = True #check_val * 100 >= bBoxSizeCheck 
            aspectRatioCheck = (w < 2 * h) & (h < 1.5 * w)
            ret, thresh = cv2.threshold(heatmap_crop.astype(np.uint8), 0, 1,cv2.THRESH_BINARY)
            bBoxIsFilledCheck = (np.sum(thresh) >= 0.7*check_val)
            
            
            if ((local_max > threshold) & bBoxIsFilledCheck & bBoxBigEnough & aspectRatioCheck):
                cv2.rectangle(image,(x_for_bBox,y_for_bBox),(x_for_bBox+150,y_for_bBox+100),(0,0,255),2)
        return image


    @staticmethod
    def load(filename="svc.model"):
        detector = CarDetector()
        detector.clf = joblib.load(filename) 
        return detector


    @staticmethod
    def from_dirs(pos_dir, neg_dir, save=True):
        """
        creates classifier by reading all training data directly from directories
        """
        features = []
        labels = []
        for filename in glob.iglob(pos_dir + '/**/*.png', recursive=True):
            features, labels = FeatureExtractor.add_feature(features, labels, filename, 1)
   
        for filename in glob.iglob(neg_dir + '/**/*.png', recursive=True):
            features, labels = FeatureExtractor.add_feature(features, labels, filename, 0)

        detector = CarDetector()
        features, labels = shuffle(features, labels, random_state=0)
        normalized_features = FeatureExtractor.normalize_features(features)
        detector.train(normalized_features, labels)
        if save:
            detector.save()
        return detector

    # END Static Methods
