import numpy as np
from skimage.feature import hog
import skimage.restoration
#from scipy.ndimage.interpolation import rotate
from skimage.transform import rotate
#from skimage.util import random_noise
from skimage import filters
import scipy.misc
from PIL import Image

IMAGE_SIZE_X = 20
IMAGE_SIZE_Y = IMAGE_SIZE_X
TEST_SET_SHARE = 0.9
MODEL_DIR = "output/1000/2017_04_26_20.49_chars74k-lite"

def normalizeNpArray(npArray):
    train_images_a = npArray
    train_images_a = train_images_a.astype(np.float32)
    for i in range(len(train_images_a)):
        mean = np.mean(train_images_a[i])
        #variance = np.var(train_images_a[i]) / 255
        for g in range(len(train_images_a[i])):
            for h in range(len(train_images_a[i][g])):
                a = train_images_a[i][g][h] - mean
                a = a/(255)
                train_images_a[i][g][h] = a
    """
    variance = np.var(train_images_a)
    for i in range(len(train_images_a)):
        for g in range(len(train_images_a[i])):
            for h in range(len(train_images_a[i][g])):
                a = train_images_a[i][g][h]
                a = a/variance
                train_images_a[i][g][h] = a
    """
    return train_images_a

def pre_process_single_img(img):
    img_y = img
    #img_y = skimage.restoration.denoise_bilateral(img_y, multichannel=False)
    #img_y = (img_y / 255.).astype(np.float32)
    #_, img_y = hog(img_y, orientations=8, pixels_per_cell=(2, 2),
    #    cells_per_block=(1, 1), visualise=True)
    #img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
    #img_y = skimage.filters.sobel(img_y)
    return img_y

def sobelFilter(img):
    return filters.sobel(img)

def augmentData(img):
    augmentedImages = [sobelFilter(img)]
    augmentedImages = [img]
    augmentedImages.append(invertImage(img))
    for imgNumber in range(0):
        for i in range(10, 60, 20):
            image = augmentedImages[imgNumber]
            augmentedImages.append(rotatePicture(image, i))
            augmentedImages.append(rotatePicture(image, 360-i))
    return augmentedImages

def invertImage(img):
    for g in range(len(img)):
        for h in range(len(img[g])):
            a = 255 - img[g][h]
            img[g][h] = a
    return img

def rotatePicture(img, degrees):
    rotated = rotate(img, degrees)
    return rotated