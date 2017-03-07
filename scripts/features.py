import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

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
