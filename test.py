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
import imageProc as ip

MODEL_DIR = ip.MODEL_DIR

def main():
    test()

def add_dimension(img):
    return img.reshape(img.shape + (1,))

def test():
    directory = ip.DATA_SET

    # Restore session and variables/nodes/weights
    session = tf.Session()
    meta_file = os.path.join(MODEL_DIR, "save.ckpt.meta")
    saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join(MODEL_DIR)
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

    # Load the test dataset.numberOfImages = len(file_names)
    test_images, test_labels = ip.load_data(test=True, augment=False)
    #test_images = load_test_data_as_numpy_array(test_data_dir)

    # Transform the images, just like we did with the training set.
    test_images = [ip.pre_process_single_img(image) for image in test_images]

    # Add dimension for tensorflow
    test_images = [add_dimension(image) for image in test_images]
    test_images = np.array(test_images)
    test_images = ip.normalizeNpArray(test_images)

    # Create a graph to hold the model.
    graph = tf.get_default_graph()

    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, ip.IMAGE_SIZE_X, ip.IMAGE_SIZE_Y, 1])

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
        predicted_image.shape = (ip.IMAGE_SIZE_X, ip.IMAGE_SIZE_Y);
        ip.save_numpy_array_as_image(predicted_image, save_dir, '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1


def load_test_data_as_numpy_array(data_dir):
    images = []
    for filename in glob.glob(data_dir + "/*.ppm"):
        images.append(skimage.data.imread(filename))  # Loads the images as a list of numpy arrays

    return images

main()