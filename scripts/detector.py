from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
#from scipy.ndimage import imread
import numpy as np
import cv2
import glob

class CarDetector:

    def __init__(self):
        self.clf = svm.SVC(kernel = "linear")

    def train(self, train_data, train_labels):
        self.clf.fit(train_data, train_labels)
        return True

    def is_car(self, image):
        image_is_car = self.clf.predict(image)
        if (image_is_car):
            return True
        return False

    def accuracy(self, test_data, test_labels):
        return self.clf.score(test_data, test_labels)

    def save(self, filename="svc.model"):
        joblib.dump(self.clf, filename) 

    def load(self, filename="svc.model"):
        self.clf = joblib.load(filename) 

    def detect(self, image, object_size=(32,32), stride=8):
        X,Y,Z = image.shape
        horizontal_steps = int(X / object_size[0])
        vertical_steps = int(Y / object_size[1])

        for i in range(0,X-object_size[0], stride):
            for j in range(int(Y/3),Y-object_size[1], stride):
                x_start = i
                y_start = j
                x_end = i+object_size[0]
                y_end = j+object_size[1]
                img_part = image[x_start:x_end, y_start:y_end, :]

                features = CarDetector.get_image_features(img_part).reshape(1, -1)
                if (self.is_car(features)):
                    cv2.rectangle(image, (x_start,y_start), (x_end,y_end), (255,0,0))
        return image;


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
            features.append(CarDetector.get_image_features(image))
            labels.append(1)
        for filename in glob.iglob(neg_dir + '/**/*.png', recursive=True):
            #print("\t- file {0} ".format(filename))
            image = cv2.imread(filename)
            features.append(CarDetector.get_image_features(image))
            labels.append(0)
        detector = CarDetector()
        normalized_features = CarDetector.normalize_features(features)
        detector.train(normalized_features, labels)
        return detector

    @staticmethod
    def get_image_features(image, size = (32,32), orient=9, pixels_per_cell=8, cells_per_block=2):
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized_feature_image = cv2.resize(feature_image, size)
        resized_gray_feature_image = cv2.resize(gray_feature_image, size)
        color_feature_vector = resized_feature_image.ravel()

        hog_features_vector = hog(resized_gray_feature_image, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=True)
        feature_vector = np.concatenate((color_feature_vector, hog_features_vector), axis=0)
        return feature_vector

    @staticmethod
    def normalize_features(features):
        #X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        features_scaler = StandardScaler().fit(features)
        # Apply the scaler to X
        scaled_features = features_scaler.transform(features)
        return scaled_features

                   
def display_image(img1, title1 = "Image"):
    fig = plt.figure()
    a=fig.add_subplot(1,1,1)
    imgplot = plt.imshow(img1)
    a.set_title(title1)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(0, 0, 640, 360)
    plt.show()

def test():
    detector = CarDetector.load()
    image = cv2.imread("./test_images/test1.jpg")
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected = detector.detect(imageRGB)
    display_image(detected)