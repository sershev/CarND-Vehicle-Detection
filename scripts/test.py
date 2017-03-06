from detector import *

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

def draw_rectangles(image, rectangles, color=(255,0,0)):
    for scaled_bBoxes in rectangles:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for rect in scaled_bBoxes:        
            cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), color, 2)
    return image


def test(image_name):
    detector = CarDetector.load()
    image = cv2.imread("./test_images/" + image_name)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bBoxes = detector.detect_multiscale(image, scale=1.5, min_sliding_window=(32,32), max_sliding_window=(256,256))

    heatmap = CarDetector.get_heatmap(bBoxes, image.shape[0:2])

    print(np.where(heatmap>0))
    display_heatmap(heatmap)

    output = draw_rectangles(image, bBoxes)
    detectedRGB = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    display_image(detectedRGB)


test("test1.jpg")
test("test2.jpg")
test("test3.jpg")
test("test4.jpg")