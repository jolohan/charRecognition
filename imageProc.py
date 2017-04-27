import numpy as np
from skimage.feature import hog
import skimage.restoration
#from scipy.ndimage.interpolation import rotate
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
#from skimage.util import random_noise
from skimage import filters
import skimage.data
import scipy.misc
from PIL import Image
import random
import os

IMAGE_SIZE_X = 20
IMAGE_SIZE_Y = IMAGE_SIZE_X
TEST_SET_SHARE = 0.2
MODEL_DIR = "output/10000/2017_04_27_01.03_chars74k-lite"
DATA_SET = "chars74k-lite"

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

def denoise(img):
    return skimage.restoration.denoise_tv_bregman(img, 0.9)

def pre_process_single_img(img):
    img_y = img
    #img_y = skimage.restoration.denoise_bilateral(img_y, multichannel=False)
    #img_y = (img_y / 255.).astype(np.float32)
    #_, img_y = hog(img_y, orientations=8, pixels_per_cell=(2, 2),
    #    cells_per_block=(1, 1), visualise=True)
    #img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
    #img_y = skimage.filters.sobel(img_y)
    return sobelFilter(img_y)

def sobelFilter(img):
    return filters.sobel(img)

def lbp(img):
    return local_binary_pattern(img, 5, 2, 'nri_uniform')

def augmentData(img, augment):
    #augmentedImages = [sobelFilter(img)]
    augmentedImages = [img]
    if not augment:
        return augmentedImages
    else:
        augmentedImages.append(invertImage(img))
        for imgNumber in range(2):
            image = augmentedImages[imgNumber]
            for g in range(4):
                degrees = int(round(random.randrange(0, 30)))
                augmentedImages.append(rotatePicture(image, degrees))
                augmentedImages.append(rotatePicture(image, 360-degrees))
            #for i in range(10, 60, 20):"""

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

def load_data(test, augment):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    data_dir = os.path.join("datasets", DATA_SET)
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []

    i = 0
    print("Start loading of ", len(directories), " image directories")
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        numberOfImages = len(file_names)
        numberOfTestImages = int(round(numberOfImages*TEST_SET_SHARE))
        start = numberOfTestImages
        stop = numberOfImages
        if (test):
            start = 0
            stop = numberOfTestImages
        for g in range(start, stop):
            f = file_names[g]
            #images.append(pre_process_single_img(skimage.data.imread(f)))
            image = skimage.data.imread(f)
            """if (test):
                for i in range(len(test_images)):
                    image = ip.denoise(test_images[i])
                    test_images[i] = image
                    """
            augmentedImages = augmentData(image, augment)
            for img in augmentedImages:
                images.append(img)
                labels.append(i)
        """for g in range(numberOfTestImages, numberOfImages):
            f = file_names[g]
            image = Image.open(f)"""

        print("Loaded directory number ", i)
        i += 1
    print("Loaded %i pics" % len(images))
    return images, labels

def save_numpy_array_as_image(array, save_dir, filename):
    #Rescale to 0-255 and convert to uint8
    to_add = array.min()
    array = array - to_add
    rescaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)
    #rescaled = (array*255).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(save_dir + filename)