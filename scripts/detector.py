from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
#from scipy.ndimage import imread
import random
import numpy as np
import cv2
import glob

#work around because of strange behaviour of StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt

class CarDetector:

    def __init__(self):
        self.clf = svm.SVC(kernel = "rbf")

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

        for i in range(0,X-object_size[0], stride):
            for j in range(int(Y/2),Y-object_size[1]-20, stride):
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
        print(min_sliding_window[0], max_sliding_window[0])
        while ((min_sliding_window[0] <= max_sliding_window[0]) & (min_sliding_window[1] <= max_sliding_window[1])):
            stride = (int)(min_sliding_window[0]/2)
            bBoxes = self.detect(image, object_size=min_sliding_window, stride=stride)
            all_window_rect_points.append(bBoxes)
            min_sliding_window = (int(min_sliding_window[0] * scale), int(min_sliding_window[1] * scale))
        return all_window_rect_points

    @staticmethod
    def get_heatmap(bBoxes, shape):

        heatmap = np.zeros(shape)
        for scaled_bBoxes in bBoxes:
            for bBox in scaled_bBoxes:
                x_start = bBox[0]
                y_start = bBox[1]
                x_end = bBox[2]
                y_end = bBox[3]
                heatmap[y_start:y_end, x_start:x_end] = heatmap[y_start:y_end, x_start:x_end] + 1
        return heatmap

    @staticmethod
    def load(filename="svc.model"):
        detector = CarDetector()
        detector.clf = joblib.load(filename) 
        return detector


    @staticmethod
    def from_dirs(pos_dir, neg_dir):
        """
        creates classifier by reading all training data directly from directories
        """
        features = []
        labels = []
        for filename in glob.iglob(pos_dir + '/**/*.png', recursive=True):
            #print("\t- file {0} ".format(filename))
            image = cv2.imread(filename)
            features.append(FeatureExtractor.get_image_features(image))
            labels.append(1)
        for filename in glob.iglob(neg_dir + '/**/*.png', recursive=True):
            #print("\t- file {0} ".format(filename))
            image = cv2.imread(filename)
            features.append(FeatureExtractor.get_image_features(image))
            labels.append(0)
        detector = CarDetector()
        normalized_features = FeatureExtractor.normalize_features(features)
        detector.train(normalized_features, labels)
        return detector



class FeatureExtractor:

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod
    def get_image_features(image, size = (32,32), orient=9, pixels_per_cell=8, cells_per_block=2):
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized_feature_image = cv2.resize(feature_image, size)
        resized_feature_image_hist = FeatureExtractor.color_hist(resized_feature_image)
        resized_gray_feature_image = cv2.resize(gray_feature_image, size)
        color_feature_vector = resized_feature_image_hist

        hog_features_vector = hog(resized_gray_feature_image, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)
        feature_vector = np.concatenate((color_feature_vector, hog_features_vector), axis=0)
        return feature_vector

    @staticmethod
    def normalize_features(features):
        # X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        features_scaler = StandardScaler().fit(features)
        # Apply the scaler to X
        scaled_features = features_scaler.transform(features)
        return scaled_features
