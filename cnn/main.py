from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

import tensorflow as tf
from Model.ModelBuilding import create_model
import cv2
import numpy as np


def normalized(rgb):
    norm = np.zeros(rgb.shape, np.float32)

    b, g, r = cv2.split(rgb)

    norm[:, :, 0] = cv2.equalizeHist(b)
    norm[:, :, 1] = cv2.equalizeHist(g)
    norm[:, :, 2] = cv2.equalizeHist(r)

    return norm


def binarylab(labels, width, height, nblbl):
    x = np.zeros([height, width, nblbl])
    for i in range(height):
        for j in range(width):
            x[i, j, labels[i][j]] = 1
    return x


def prep_data(base_path, from_path, width, height):
    train_data = []
    train_label = []

    img_path = os.path.join(base_path, from_path, "src")
    lbl_path = os.path.join(base_path, from_path, "labels")

    trainImgs = sorted([f for f in os.listdir(img_path)])
    labelImgs = sorted([f for f in os.listdir(lbl_path)])

    print(len(trainImgs))

    assert len(trainImgs) == len(labelImgs)

    for i in range(len(trainImgs)):
        train_data.append(np.rollaxis(normalized(cv2.imread(os.path.join(img_path, trainImgs[i]))), 2))
        train_label.append(binarylab(cv2.imread(os.path.join(lbl_path, labelImgs[i]))[:, :, 0], width, height, nblbl))

    return np.array(train_data), np.array(train_label), trainImgs, labelImgs


def get_images_for_tests(image_path):
    img_names = sorted([f for f in os.listdir(image_path)])

    img_files = []

    for i in range(len(img_names)):
        img_files.append(np.rollaxis(normalized(cv2.imread(os.path.join(image_path, img_names[i]))), 2))

    return img_names, np.array(img_files)


def train_model(width, height, nblbl, dataset_path, weights_filepath):
    data_shape = width * height
    np.random.seed(1337)  # for reproducibility

    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height)
    train_label = np.reshape(train_label, (len(train_data), data_shape, nblbl))

    class_weighting = [2.0, 4.61005688, 10.03329372, 5.45229053]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        nb_epoch = 100
        batch_size = 3

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, class_weight=class_weighting, shuffle=True)

        autoencoder.save_weights(weights_filepath)


def test_model(width, height, nblbl, test_images_path, weights_filepath, prediction_output_path):
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]

    label_colours = np.array([Void, Sky, Building, Vegetation])

    src_path, test_data = get_images_for_tests(test_images_path)

    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    for i in range(len(test_data)):
        output = autoencoder.predict_proba(test_data[i:i+1])
        pred = visualize(np.argmax(output[0], axis=1).reshape((height, width)), label_colours, nblbl)
        cv2.imwrite(os.path.join(prediction_output_path, src_path[i]), pred)


def visualize(temp, label_colours, nblbl):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, nblbl):
        r[temp == l] = label_colours[l, 2]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 0]

    rgb = cv2.merge((b, g, r))
    return rgb


if __name__ == '__main__':
    width = 360
    height = 360
    nblbl = 4
    nb_epoch = 100
    dataset_path = "./dataset/"
    test_images_path = "./test_images/"

    weigths_filepath = "./weigths_svf_ep%d_s%dx%d.hdf5" % (nb_epoch, width, height)

    #train_model(width, height, nblbl, dataset_path, weigths_filepath)
    test_model(width, height, nblbl, test_images_path, weigths_filepath, "./predictions")
