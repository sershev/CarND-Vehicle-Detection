from detector import *
#debug
#np.set_printoptions(threshold=np.nan)

def display_image(img1, title1 = "Image"):
    fig = plt.figure()
    a=fig.add_subplot(1,1,1)
    imgplot = plt.imshow(img1)
    a.set_title(title1)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(0, 0, 640, 360)
    plt.show()

def display_heatmap(heatmap, title1 = "Image"):
    fig = plt.figure()
    a=fig.add_subplot(1,1,1)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    a.set_title(title1)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(0, 0, 640, 360)
    plt.show()

def draw_rectangles(image, rectangles):
    for scaled_bBoxes in rectangles:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for rect in scaled_bBoxes:        
            cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), color, 2)
    return image


def compare_before_after(img1, img2, title1 = "Before", title2 = "After"):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(img1)
    a.set_title(title1)
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(img2)
    a.set_title(title2)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(0, 0, 640, 360)
    plt.show()


def test(image_name):
    detector = CarDetector.load()
    image = cv2.imread("./test_images/" + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bBoxes = detector.detect_multiscale(image, scale=1.5, min_sliding_window=(64,64), max_sliding_window=(128,128))

    #image_with_rectangles = draw_rectangles(image, bBoxes)
    #display_image(image_with_rectangles, "Detect multiscale result.")

    heatmap = CarDetector.get_heatmap(bBoxes, image.shape[0:2])
    display_heatmap(heatmap)
    #print (heatmap)
    
    contours = CarDetector.get_countours_of_heatmap(heatmap)
    output = CarDetector.heatmap_contours_to_bBoxes(image, contours, heatmap)
    
    display_image(output)

#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput

#with PyCallGraph(output=GraphvizOutput()):
test("test1.jpg")
test("test2.jpg")
test("test3.jpg")
test("test4.jpg")

