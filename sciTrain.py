# Import `datasets` from `sklearn`
from sklearn import datasets
import imageProc as ip
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm

# Load in the `digits` data
images, labels = ip.load_data(test=False, augment=False) ## needs to be changed
#images = [ip.pre_process_single_img(image) for image in images]
images_and_labels = zip(images, labels)
number_chars = len(np.unique(labels))

# Create a regular PCA model
pca = PCA(n_components=3)

# Fit and transform the data to the model
r_images = [np.resize(image,(ip.IMAGE_SIZE_X, ip.IMAGE_SIZE_Y)).flatten() for image in images]
reduced_data_pca = pca.fit_transform(r_images)

# Inspect the shape
reduced_data_pca.shape

# Print out the data
#print(reduced_data_pca)

def plotting(images, labels):
    # Figure size (width, height) in inches
    fig = plt.figure(figsize=(6, 6))

    # Adjust the subplots
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # For each of the 64 images
    for i in range(64):
        # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        # Display an image at the i-th position
        ax.imshow(images[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(labels[i]))

    # Show the plot
    plt.show()

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
colors += ['aliceblue', 'antiquewhite', 'aqua', 'coral', 'brown', 'chocolate', 'bisque', 'azure',
           'burlywood', 'chartreuse']
colors += ['beige', 'crimson', 'cornsilk', 'ivory', 'khaki', 'linen']

""""
for i in range(len(colors)):
    x = reduced_data_pca[:, 0][labels[:] == i]
    y = reduced_data_pca[:, 1][labels[:] == i]
    plt.scatter(x, y, c=colors[i])"""

for i in range(len(colors)):
    x = []
    y = []
    for g in range(len(reduced_data_pca)):
        if (labels[g] == i):
            x.append(reduced_data_pca[g][0])
            y.append(reduced_data_pca[g][1])

    plt.scatter(x, y, c=colors[i])


plt.legend(np.unique(labels), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
#plt.show()

lin_clf = svm.LinearSVC()
lin_clf.fit(reduced_data_pca, labels)
test_images, test_labels = ip.load_data(test=False, augment=False) ## needs to be changed
r_images = [np.resize(image,(ip.IMAGE_SIZE_X, ip.IMAGE_SIZE_Y)).flatten() for image in test_images]
reduced_data_pca = pca.fit_transform(r_images)
pred_labels = []
for i in range(len(reduced_data_pca)):
    dec = lin_clf.decision_function(reduced_data_pca[i].reshape(1, -1))
    decMax = dec[0]
    decMaxIndex = 0
    for g in range(1, len(dec)):
        if (dec[g] > decMax):
            decMax = dec[g]
            decMaxIndex = g
    label = decMaxIndex
    pred_labels.append(label)

if (len(test_labels) != len(pred_labels)):
    print("Length of test labels != length og predicted and accuracy is not correctly calculated")
    print("Length of test labels: ", len(test_labels))
    print("Lenght of predicted: ", len(pred_labels))
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, pred_labels)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}". format(accuracy))