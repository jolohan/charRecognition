import os
#import cv2
import skimage.data
import skimage.transform
import skimage.exposure as exposure
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import glob
import time
import imageProc as ip
from PIL import Image

DISPLAY_FREQUENCY = 50
CONTINUE_TRAINING_ON_MODEL = False

TRAINING_DATA_SET = "chars74k-lite"

# Network paramters
TRAINING_NUMBER = 1000
LEARNING_RATE = 0.01
NUMBER_OF_HIDDEN_NODES_1 = 100
NUMBER_OF_HIDDEN_NODES_2 = 100
#NUMBER_OF_HIDDEN_NODES_3 = 52
NUMBER_OF_LOGITS = 26

def main():
    start_time = time.time()
    train_loss_a, display_iter = train()
    display_train_loss(display_iter, train_loss_a)
    end_time = time.time()
    print ("Total time elapsed: ", (end_time-start_time))

def display_train_loss(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.ylabel('Train loss')
    plt.xlabel('Number of iterations')
    plt.grid(True)
    plt.show()

def load_test_data_as_numpy_array(data_dir):
    images = []
    for filename in glob.glob(data_dir+"/*.ppm"):
        images.append(skimage.data.imread(filename)) #Loads the images as a list of numpy arrays

    return images

def save_images_and_labels_to_imagefile(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image, cmap="gray")
    #plt.show()
    plt.savefig('labels_and_corresponding_images.png')

def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

def save_model(sess):
    if not os.path.exists("output"): os.makedirs("output")
    dirname = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M') + "_" + str(TRAINING_DATA_SET)
    filename = str(TRAINING_NUMBER)
    os.makedirs(os.path.join("output", filename, dirname))

    saver = tf.train.Saver()
    saver.save(sess, os.path.abspath(os.path.join("output", filename, dirname, "save.ckpt")))
    # `save` method will call `export_meta_graph` implicitly.
    # you will get saved graph files:my-model.meta

def add_dimension(img):
    return img.reshape(img.shape + (1,))

def normalizeNpArray(npArray):
    train_images_a = npArray
    train_images_a = train_images_a.astype(np.float32)
    mean = np.mean(train_images_a)
    for i in range(len(train_images_a)):
        for g in range(len(train_images_a[i])):
            for h in range(len(train_images_a[i][g])):
                a = train_images_a[i][g][h] - mean
                a = a/255
                train_images_a[i][g][h] = a
    return train_images_a

def train():
    # Load training and testing datasets.

    train_images, labels = ip.load_data(test=False, augment=True)

    print("Unique Labels: {0}\nTotal Train Images: {1}".format(len(set(labels)), len(train_images)))

    # display_images_and_labels(train_images, labels)

    # display_label_images(train_images, 27)

    # Resize images = not needed
    print("Pre-processing pics")
    i = 0
    #train_images = [ip.pre_process_single_img(image) for image in train_images]
    print("Done with pre-processing")

    #save_images_and_labels_to_imagefile(train_images, labels)

    # Add dimension for tensorflow
    train_images = [add_dimension(image) for image in train_images]

    labels_a = np.array(labels)
    print("labels_a:", labels_a)
    train_images_a = np.array(train_images)
    train_images_a = ip.normalizeNpArray(train_images_a)

    print("labels: ", labels_a.shape, "\nTrain images: ", train_images_a.shape)

    """if CONTINUE_TRAINING_ON_MODEL:
        # Restore session and variables/nodes/weights
        session = tf.Session()
        meta_file = os.path.join("output", MODEL_DIR, "save.ckpt.meta")
        saver = tf.train.import_meta_graph(meta_file)

        checkpoint_dir = os.path.join("output", MODEL_DIR)
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        graph = tf.get_default_graph()
    else:"""
    # Create a graph to hold the model.
    graph = tf.Graph()

    # Create model in the graph.
    with graph.as_default():
        # Placeholders for inputs and labels.
        if CONTINUE_TRAINING_ON_MODEL:
            images_ph = graph.get_tensor_by_name("images_ph:0")
            labels_ph = graph.get_tensor_by_name("labels_ph:0")
        else:
            images_ph = tf.placeholder(tf.float32, [None, ip.IMAGE_SIZE_X, ip.IMAGE_SIZE_Y, 1],
                                   name="images_ph")
            labels_ph = tf.placeholder(tf.int32, [None], name="labels_ph")

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        if CONTINUE_TRAINING_ON_MODEL:
            weights_0 = tf.global_variables()[0]
            biases_0 = tf.global_variables()[1]
            weights_1 = tf.global_variables()[2]
            biases_1 = tf.global_variables()[3]
            weights_2 = tf.global_variables()[4]
            biases_2 = tf.global_variables()[5]

            hidden1 = tf.nn.relu(tf.matmul(images_flat, weights_0) + biases_0)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_1) + biases_1)
            logits = tf.nn.relu(tf.matmul(hidden2, weights_2) + biases_2)

            loss = graph.get_tensor_by_name("Lossy:0")
            train = graph.get_operation_by_name("Adam")
        else:
            hidden1 = tf.contrib.layers.fully_connected(images_flat, NUMBER_OF_HIDDEN_NODES_1, tf.nn.relu)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, NUMBER_OF_HIDDEN_NODES_2, tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(hidden2, NUMBER_OF_LOGITS, tf.nn.relu)

            # Define the loss function.
            # Cross-entropy is a good choice for classification.
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph),
                                  name="Lossy")
            # Create training op.
            adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name="Adam")
            train = adam.minimize(loss)
            #print(train)

        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)

        """print("images_flat: ", images_flat)
        print("logits: ", logits)
        print("loss: ", loss)
        print("predicted_labels: ", predicted_labels)
        print("images_ph: ", images_ph)"""

        if not CONTINUE_TRAINING_ON_MODEL:
            # And, finally, an initialization op to execute before training.
            init = tf.global_variables_initializer()

            # Create a session to run the graph we created.
            session = tf.Session(graph=graph)

            # First step is always to initialize all variables.
            # We don't care about the return value, though. It's None.
            _ = session.run([init])

        #The actual training
        start = time.time()
        loss_value = None
        train_loss_a = []
        display_iter = []

        for i in range(1, TRAINING_NUMBER+1):
            _, loss_value = session.run([train, loss],
                                        feed_dict={images_ph: train_images_a, labels_ph: labels_a})
            if i % DISPLAY_FREQUENCY == 1:
                print("Iter: " + str(i) +", Loss: ", loss_value, ", Time elapsed: ", time.time()-start)
            train_loss_a.append(loss_value)
            display_iter.append(i)

        print("Iter: " + str(TRAINING_NUMBER) + ", Loss: ", loss_value, ", Time elapsed: ", time.time() - start)

       # Save session
        save_model(session)
        # Close the session. This will destroy the trained model.
        session.close()
        print("Model saved.")

        return train_loss_a, display_iter

main()