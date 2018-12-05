import os
import cv2
import numpy as np
import pandas as pd
def preprocess_images(data_matrix, directory):
    """
    This function preprocesses the images refered to in the data matrix by loading them into a file.
    :param data_matrix:
    :param directory:
    :return:
    """
    images = []
    image_num = 0
    for ind,i in enumerate(data_matrix[:, 1]):
        path = os.path.join(directory,str(i) + ".jpg")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images.append(img)
    return images

NUM_CLASSES = 4

def build_histograms(preprocessed_images, num_channels):
    """
    This function builds historams from preprocessed data.
    :param preprocessed_images:
    :param num_channels:
    :return:
    """
    histogram_processed_train = np.zeros((len(preprocessed_images), num_channels ** 3))
    for i, img in enumerate(preprocessed_images):
        # chans = cv2.split(image)
        colors = ("b", "g", "r")
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [num_channels, num_channels, num_channels], [0, 256, 0, 256, 0, 256])
        histogram_processed_train[i] = hist.flatten()
    return histogram_processed_train


def trainModelStack(models, combiner, datas, labels):
    """
    Trains model stack :)
    We aren't checking for valid input oops.
    :param models: A list of trained models
    :param combiner: An untrained model for use in combining the models
    :param datas: A list of the data that each model is trained on
    :param labels: The labels for the training data
    :return:
    """

    combined_probabilities = np.zeros((labels.size, NUM_CLASSES*len(models)))
    probs = list(map(lambda x: x[0].predict_proba(x[1]), zip(models, datas)))

    for i in range(labels.size):
        x = np.array([])
        for prob in probs:
            x = np.append(x, prob[i])
        combined_probabilities[i] = x

    combiner.fit(combined_probabilities, labels)


def predictModelStack(models, combiner, datas):
    size = datas[0].shape[0]
    combined_probabilities = np.zeros((size, 4*len(models)))
    probs = list(map(lambda x: x[0].predict_proba(x[1]), zip(models, datas)))

    for i in range(size):
        x = np.array([])
        for prob in probs:
            x = np.append(x, prob[i])
        combined_probabilities[i] = x
    predictions = combiner.predict(combined_probabilities)
    return predictions


def predictions_to_csv(predictions, fileName):
    labels = ['Id', 'Category']
    df = pd.DataFrame.from_records(enumerate(predictions), columns=labels)
    df.to_csv(fileName, mode='w', index=False)