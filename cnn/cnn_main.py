from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

import tensorflow as tf
from .Model.ModelBuilding import create_model
import cv2
import numpy as np
from tools.FileManager import FileManager


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


def prep_data(base_path, from_path, width, height, nblbl, magentize=True):
    train_data = []
    train_label = []

    img_path = os.path.join(base_path, from_path, "src")
    lbl_path = os.path.join(base_path, from_path, "labels")

    train_imgs = sorted([f for f in os.listdir(img_path)])
    label_imgs = sorted([f for f in os.listdir(lbl_path)])

    assert len(train_imgs) == len(label_imgs)

    color = (255, 0, 255)
    mask = get_void_mask(height, width)
    idx = mask > 0

    for i in range(len(train_imgs)):
        src_img = normalized(FileManager.LoadImage(train_imgs[i], img_path))
        lbl_img = FileManager.LoadImage(label_imgs[i], lbl_path)

        if magentize:
            src_img = colorize_void(idx, color, src_img)

        train_data.append(np.rollaxis(src_img, 2))
        train_label.append(binarylab(lbl_img[:, :, 0], width, height, nblbl))

    return np.array(train_data), np.array(train_label), train_imgs, label_imgs


def get_void_mask(width, height):
    mask = np.zeros((height, width, 1), np.uint8)
    cv2.circle(mask, (width / 2, height / 2), width / 2, (255, 255, 255), -1)
    mask = cv2.bitwise_not(mask)
    return mask


def colorize_void(idx, color, src_img):
    b, g, r = cv2.split(src_img)
    b = b.reshape((b.shape[0], b.shape[1], 1))
    g = g.reshape((g.shape[0], g.shape[1], 1))
    r = r.reshape((r.shape[0], r.shape[1], 1))
    b[idx] = color[0]
    g[idx] = color[1]
    r[idx] = color[2]
    return cv2.merge((b, g, r))


def get_images_for_tests(image_path, width, height, magentize=True):
    img_names = sorted([f for f in os.listdir(image_path)])

    img_files = []

    color = (255, 0, 255)
    mask = get_void_mask(width, height)
    idx = mask > 0

    for i in range(len(img_names)):
        src_img = normalized(FileManager.LoadImage(img_names[i], image_path))
        if magentize:
            src_img = colorize_void(idx, color, src_img)
        img_files.append(np.rollaxis(src_img, 2))

    return img_names, np.array(img_files)


def train_model(width, height, nblbl, dataset_path, weights_filepath):
    data_shape = width * height
    np.random.seed(1337)  # for reproducibility

    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, nblbl)
    train_label = np.reshape(train_label, (len(train_data), data_shape, nblbl))

    class_weighting = [2.0, 4.61005688, 10.03329372, 5.45229053]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        nb_epoch = 100
        batch_size = 2

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, class_weight=class_weighting, shuffle=True)

        autoencoder.save_weights(weights_filepath)


def test_model(width, height, nblbl, test_images_path, weights_filepath, prediction_output_path):
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [255, 255, 255]

    label_colours = np.array([Void, Sky, Building, Vegetation])

    src_path, test_data = get_images_for_tests(test_images_path, width, height)

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
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = cv2.merge((r, g, b))
    return rgb


def main(width, height):
    nblbl = 4
    nb_epoch = 100
    dataset_path = "./cnn/dataset/"
    test_images_path = "./cnn/test_images/"

    weigths_filepath = "./cnn/weigths_svf_ep%d_s%dx%d_magentavoid.hdf5" % (nb_epoch, width, height)

    #train_model(width, height, nblbl, dataset_path, weigths_filepath)
    test_model(width, height, nblbl, test_images_path, weigths_filepath, "./predictions")
