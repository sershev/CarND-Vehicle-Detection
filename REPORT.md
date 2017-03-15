**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier (e.g. Linear SVM classifier).
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car_example.png
[image2]: ./output_images/new_car_and_hog.png
[image22]: ./output_images/new_noncar_and_hog.png
[image3]: ./output_images/detect_multiscale_result.png
[image4]: ./output_images/example1.png
[image5]: ./output_images/example2.png
[image6]: ./output_images/example3.

[image55]: ./output_images/heatmap_output1.png
[image66]: ./output_images/heatmap_output2.png
[image77]: ./output_images/heatmap_output4.png

[image7]: ./examples/output_bboxes.png
[video1]: ./out_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `scripts/features.py` in static `get_image_features()` method.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle non vehicle comparison.][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `GREY` single channel color space and HOG parameters of `orientations=18`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

####Update
 After experimenting for a while I cahnged:
 * using YUV colorspace 3 channel for HOG features, instead `GREY` single channel
 * orient to 24, which provided more accurate results.



![Car and hog image][image2]  ![Non car and hog image.][image22]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choosed one which provided suitable tradeoff between speed and accuracy (subjective feelings).
Better it would be to run a benchmark for different parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a an SVM using rbf kernel. The code for training is in `scripts/detector.py` in `CarDetector` class (in `from_dirs()` for dataset). I also implemented `save()` and `load()` to methods to make the classifier usable later.


####Update
 * Kernel changed from `rbf` to `linear`
 * For some reason I get very bad results trying (3 channel HOG with rbf kernel) or (one (gray) channel HOG with linear kernel)
 * but 3 channel HOG with linear kernel works well.
 * svc decision_function is used to require the classiefier to be more confident to remove false positives. (I tryed different values and ended with 0.8 where 0.0 is the threshold by default)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search in the bottom half of the image (with a bit padding at most down, since there is a part of own car). The sliding window increeses each step by a percentage value until it becomes maximal size. For window overlapping I choose 50%. The code for this is in `scripts/detector.py` in `detect_multiscale()` method of `CarDetector` class. 

####Update
 * I decreased scale factor (125% isntead of 150%), so more different scales are used.
 * With a scale factor less than 125% the results would be even better, but the speed would be slower.
 * Window overlapping is now 66%.

As result we het image like following:

![Detect multiscale output example.][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


Ultimately I searched on two scales using GRAY 1-channel HOG features plus histograms of color from HSV space in the feature vector, which provided a nice result.  Here are some example images:

![Example of detection 1][image4]
![Example of detection 2][image5]
![Example of detection 3][image6]

I also used different options like different channels, more fetures, but the results were not so well.

####Update
 * I wrote new function `x_out_of_n_heatmaps()` in `detector.py` in `CarDetector` calss. This function now checks if some region was detected in at least x out of n frames. Erlier I checked if region was detected in all past n frames which leads to many false negatives. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used OpenCV `findContours()` and `boundingRect()` functions ( in `scripts/detector.py` in `get_countours_of_heatmap()` and `heatmap_contours_to_bBoxes()`)  to identify individual blobs in the heatmap.  Then I added some checks to make sure bounding box make sense (e.g. aspect ratio and if bounding box is well filled). I also averaged heatmaps over multiple frames.

Here's an example result showing the heatmap from a series of frames of video, and the bounding boxes then overlaid on the frame of video:

### Here are 3 frames and their corresponding heatmaps:

![Heatmap and coresponding output 1.][image55]
![Heatmap and coresponding output 2.][image66]
![Heatmap and coresponding output 3.][image77]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I replaced `np.histogram()` function from lesson by `cv2.calcHist()` since it is faster. Generally the execution speed is in such an application very important, so I tried to optimize it by changing different parameters and optimize my code on some places. 

The pipeline may fail very easy for many different types of vehicles, like microbus, police car, maybe smart car. Any other road traffic user like bycicle can't be detected by this classifier. Different light conditions (darknes, tunnel)
can break the detection. This is actually very basic pipeline.

####Update
 * It would make sense to add tracker component to have more stable results.