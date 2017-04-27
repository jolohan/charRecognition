import test
import skimage.data as data
import imageProc as ip
import numpy as np
from PIL import Image, ImageDraw

filename = "detection-2.jpg"
stepSize = 10

def main():
    image = data.imread(filename)
    im = Image.open(filename)
    detectedImage = detect(image)
    drawRectangle(im, detectedImage)

def drawRectangle(im, detectedImage):
    draw = ImageDraw.Draw(im)
    for y in range(len(detectedImage)):
        for x in range(len(detectedImage[y])):
            cell = detectedImage[y][x]
            if (cell == 1):
                #print(x,y)
                draw.rectangle((x*stepSize, y*stepSize,
                                x*stepSize+ip.IMAGE_SIZE_X, y*stepSize+ip.IMAGE_SIZE_Y))
    im.show()

def detect(image):
    detectedImage = []
    x_tiles = int(len(image[0])/stepSize)-1
    y_tiles = int(len(image)/stepSize)-1
    for i in range(y_tiles):
        detectedImage.append([])
        for g in range(x_tiles):
            detectedImage[i].append(0)
    outputs = get_outputs(image)
    for i in range(len(outputs)):
        output = outputs[i].max()
        if (output > 5):
            x , y = getXY(image, i)
            #print(x, i, x_tiles, i/x_tiles)
            detectedImage[y][x] = 1
    return detectedImage

def getXY(image, i):
    x_tiles = int(len(image[0]) / stepSize) - 1
    y_tiles = int(len(image) / stepSize) - 1
    return (i%x_tiles, int((i/x_tiles)-0.5))

#def joinSquares(image, )

def get_outputs(image):
    sub_images = load_images(image)
    sub_images = test.transform_images(sub_images)
    outputs = test.loadGraphAndTest(sub_images, whole_output=True)
    return outputs

def load_images(image):
    print("load image")
    sub_images = []
    for y in range(0, len(image), stepSize):
        for x in range(0, len(image[0]), stepSize):
            sub_image = get_image_segment(y, x, image)
            if (sub_image is False):
                pass
            else:
                sub_images.append(sub_image)

    return sub_images

def get_image_segment(y, x, image):
    end_x = x + ip.IMAGE_SIZE_X
    end_y = y + ip.IMAGE_SIZE_Y
    sub_image = []
    if ((len(image) >= end_y) and (len(image[0]) >= end_x)):
        for row in range(y, end_y):
            sub_image.append([])
            for col in range(x, end_x):
                sub_image[row-y].append(image[row][col])

        return np.array(sub_image)
    return False

main()