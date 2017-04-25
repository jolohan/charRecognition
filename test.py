import os
import skimage.data
import skimage.filters
import skimage.transform
import skimage.exposure as exposure
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import glob
#import cv2

MODEL_DIR = "output/500/2017_04_25_16.13_chars74k-lite"
TEST_DATA_SET = "chars74k-lite"

TEST_SET_SHARE = 0.1
IMAGE_SIZE_X = 20
IMAGE_SIZE_Y = IMAGE_SIZE_X

def main():
    test()

def normalizeNpArray(npArray):
    images_a = npArray
    images_a = images_a.astype(np.float32)
    for i in range(len(images_a)):
        for g in range(len(images_a[i])):
            for h in range(len(images_a[i][g])):
                a = images_a[i][g][h]/255.0
                images_a[i][g][h] = a
    return images_a

def pre_process_single_img(img):
    img_y = img
    img_y = (img_y / 255.).astype(np.float32)
    #img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
    #img_y = skimage.filters.sobel(img_y)
    return img_y

def add_dimension(img):
    return img.reshape(img.shape + (1,))

def test():
    directory = TEST_DATA_SET

    # Restore session and variables/nodes/weights
    session = tf.Session()
    meta_file = os.path.join(MODEL_DIR, "save.ckpt.meta")
    saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join(MODEL_DIR)
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

    # Load the test dataset.
    test_images, test_labels = load_data()
    for i in range(len(test_images)):
        image = pre_process_single_img(test_images[i])
        test_images[i] = image
    #test_images = load_test_data_as_numpy_array(test_data_dir)

    # Transform the images, just like we did with the training set.
    #train_images = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
    #                 for image in test_images]

    # Add dimension for tensorflow
    test_images = [add_dimension(image) for image in test_images]
    test_images = np.array(test_images)
    #test_images = normalizeNpArray(test_images)

    # Create a graph to hold the model.
    graph = tf.get_default_graph()

    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1])

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        weights_0 = tf.global_variables()[0]
        biases_0 = tf.global_variables()[1]
        weights_1 = tf.global_variables()[2]
        biases_1 = tf.global_variables()[3]
        weights_2 = tf.global_variables()[4]
        biases_2 = tf.global_variables()[5]

        #weights_0 = tf.get_variable("fully_connected/weights:0")
        #biases_0 = tf.get_variable("fully_connected/biases:0")
        #weights_1 = tf.get_variable("fully_connected_1/weights:0")
        #biases_1 = tf.get_variable("fully_connected_1/biases:0")

        #for var in tf.global_variables():
            #print(var)
            #print(var.name)

        hidden1 = tf.nn.relu(tf.matmul(images_flat, weights_0) + biases_0)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_1) + biases_1)
        logits = tf.nn.relu(tf.matmul(hidden2, weights_2) + biases_2)

        predicted_labels = tf.argmax(logits, 1)

    # Run predictions against the full test set.

    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images})[0]

    # Calculate how many matches we got.
    if (len(test_labels) != len(predicted)):
        print("Length of test labels != length og predicted and accuracy is not correctly calculated")
        print("Length of test labels: ", len(test_labels))
        print("Lenght of predicted: ", len(predicted))
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}". format(accuracy))

    # EVALUATING THE TEST
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')
    save_dir = 'output/' + directory + '/predictions_' + timestamp

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0

    for pl in predicted:
        predicted_image = test_images[i]
        predicted_image.shape = (IMAGE_SIZE_X, IMAGE_SIZE_Y);
        save_numpy_array_as_image(predicted_image, save_dir, '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1


def load_test_data_as_numpy_array(data_dir):
    images = []
    for filename in glob.glob(data_dir + "/*.ppm"):
        images.append(skimage.data.imread(filename))  # Loads the images as a list of numpy arrays

    return images

def save_numpy_array_as_image(array, save_dir, filename):
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(save_dir + filename)

def load_data():
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    data_dir = os.path.join("datasets", TEST_DATA_SET, "Testing")
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
        for g in range(int(round(numberOfImages*TEST_SET_SHARE))):
            f = file_names[g]
            #images.append(pre_process_single_img(skimage.data.imread(f)))
            images.append(skimage.data.imread(f))
            labels.append(i)

        print("Loaded directory number ", i)
        i += 1
    print("Loaded %i pics" %len(images))
    return images, labels

main()