import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    Helper Class for feature extraction
    """

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately

        # cv2.calcHist is faster than np.histogram
        cv_channel1_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[nbins], ranges=[0, 256]).astype(np.uint8).reshape(nbins,)
        cv_channel2_hist = cv2.calcHist([img], channels=[1], mask=None, histSize=[nbins], ranges=[0, 256]).astype(np.uint8).reshape(nbins,)
        cv_channel3_hist = cv2.calcHist([img], channels=[2], mask=None, histSize=[nbins], ranges=[0, 256]).astype(np.uint8).reshape(nbins,)

        #channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)[0]
        #channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)[0]
        #channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)[0]

        # Concatenate the histograms into a single feature vector
        #hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))
        cv_hist_features = np.concatenate((cv_channel1_hist, cv_channel2_hist, cv_channel3_hist))
        # Return the individual histograms, bin_centers and feature vector
        #import pdb
        #pdb.set_trace()
        return cv_hist_features

    def bin_spatial(img):
        color1 = img[:,:,0].ravel()
        color2 = img[:,:,1].ravel()
        color3 = img[:,:,2].ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def get_image_features(image_rgb, size = (32,32), orient=24, pixels_per_cell=8, cells_per_block=2, color_bins=64):
        resized_feature_image = cv2.resize(image_rgb, size)
        
        feature_image = cv2.cvtColor(resized_feature_image, cv2.COLOR_RGB2HSV)
        yuv_image = cv2.cvtColor(resized_feature_image, cv2.COLOR_RGB2YUV)
        gray_feature_image = cv2.cvtColor(resized_feature_image, cv2.COLOR_RGB2GRAY)

        resized_feature_image_hist = FeatureExtractor.color_hist(feature_image, color_bins)
        color_feature_vector = resized_feature_image_hist

        #hog_features_vector = hog(gray_feature_image, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        #               cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
        #               visualise=False, feature_vector=True)

        hog_features_vector = FeatureExtractor.get_hog_features(yuv_image, orient=orient, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

        #bin_spatial_features = FeatureExtractor.bin_spatial(feature_image)

        feature_vector = np.hstack((color_feature_vector, hog_features_vector))
        return feature_vector

    @staticmethod
    def normalize_features(features):
        # X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        features_scaler = StandardScaler().fit(features)
        # Apply the scaler to X
        scaled_features = features_scaler.transform(features)
        return scaled_features

    @staticmethod
    def add_feature(features_buffer, labels_buffer, filename, feature_class):
        image_bgr = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        features_buffer.append(FeatureExtractor.get_image_features(image_rgb))
        labels_buffer.append(feature_class)
        return features_buffer, labels_buffer



    @staticmethod
    def get_hog_features(image, orient, pixels_per_cell, cells_per_block):
        ch1 = image[:,:,0]
        ch2 = image[:,:,1]
        ch3 = image[:,:,2]

        hog1 =  hog(ch1, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)
        hog2 =  hog(ch2, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)
        hog3 =  hog(ch3, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)

        hog_features = np.hstack((hog1,hog2,hog3))

        return hog_features
