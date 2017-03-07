from sklearn import svm
from sklearn.externals import joblib
from features import FeatureExtractor
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
        ret, thresh = cv2.threshold(heatmap_u8c1,2,255,cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("Cnts: ", len(contours_hierarchy))
        return contours

    @staticmethod
    def heatmap_contours_to_bBoxes(image, contours):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            print(x,y,w,h)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        return image


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
