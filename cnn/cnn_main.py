from __future__ import absolute_import
from __future__ import print_function
import os
from itertools import izip
from tools.ImageDataGenerator import ImageDataGenerator

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

import tensorflow as tf
from .Model.ModelBuilding import create_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ntpath


def prep_data(base_path, from_path, width, height, nblbl, magentize=True, normalize=True):
    train_data = []
    train_label = []

    img_path = os.path.join(base_path, from_path, "src")
    lbl_path = os.path.join(base_path, from_path, "labels")

    idg = ImageDataGenerator(img_path, lbl_path, width, height, nblbl, normalize_img=normalize, magentize=magentize)

    i = 0
    for img in idg.image_generator():
        train_data.append(img)
        i += 1
        if i == len(idg.img_files):
            break

    i = 0
    for lbl in idg.label_generator():
        train_label.append(lbl)
        i += 1
        if i == len(idg.lbl_files):
            break

    return np.array(train_data), np.array(train_label), idg.img_files, idg.lbl_files



def get_images_for_tests(image_path, width, height, magentize=True, limit=10):
    idg = ImageDataGenerator(image_path, image_path, width, height, 4, magentize=magentize, shuffled=False)

    length = len(idg.img_files) if limit < 0 else limit

    i = 0
    for img in idg.image_generator():
        yield idg.img_files[i], img
        i += 1
        if i == length:
            break


def train_model(width, height, nblbl, dataset_path, weights_filepath):
    data_shape = width * height
    np.random.seed(1337)  # for reproducibility

    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, nblbl)
    valid_data, valid_label, _, _ = prep_data(dataset_path, "valid", width, height, nblbl)

    #class_weighting = [2.0, 4.61005688, 10.03329372, 5.45229053]  # dataset 100
    class_weighting = [1.99913108, 4.76866531,  9.54897594, 5.39499044]  # dataset 193

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        nb_epoch = 100
        batch_size = 2

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                                  class_weight=class_weighting, shuffle=True, validation_data=(valid_data, valid_label))

        autoencoder.save_weights(weights_filepath)
        graph_path = os.path.join("./cnn/weights/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(train_data),
                          len(valid_data), graph_path, data_augmentation=False)


def train_model_generators(width, height, nblbl, classes_weights, dataset_path, weights_filepath):
    class_weighting = [1.99913108, 4.76866531, 9.54897594, 5.39499044]
    batch_size = 2
    nb_epoch = 100
    sample_per_epoch = 250
    sample_val = 28

    img_path_train = os.path.join(dataset_path, "train", "src")
    lbl_path_train = os.path.join(dataset_path, "train", "labels")

    img_path_valid = os.path.join(dataset_path, "valid", "src")
    lbl_path_valid = os.path.join(dataset_path, "valid", "labels")

    idg_train = ImageDataGenerator(img_path_train, lbl_path_train, width, height, nblbl, 90, 270, rotate=True,
                                   batch_size=batch_size)
    idg_valid = ImageDataGenerator(img_path_valid, lbl_path_valid, width, height, nblbl, 90, 270, rotate=True,
                                   batch_size=batch_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        train_generator = izip(idg_train.image_batch_generator(), idg_train.label_batch_generator())
        valid_generator = izip(idg_valid.image_batch_generator(), idg_valid.label_batch_generator())

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit_generator(train_generator, samples_per_epoch=sample_per_epoch, nb_epoch=nb_epoch, verbose=1,
                                            validation_data=valid_generator, nb_val_samples=sample_val, class_weight=class_weighting)

        autoencoder.save_weights(weights_filepath)

        graph_path = os.path.join("./cnn/weigths/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(idg_train.img_files),
                          len(idg_valid.img_files), graph_path, data_augmentation=True, sample_per_epoch=sample_per_epoch,
                          nb_val_sample=sample_val)


def test_model(width, height, nblbl, test_images_path, weights_filepath, prediction_output_path):
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]

    label_colours = np.array([Void, Sky, Building, Vegetation])

    #src_path, test_data = get_images_for_tests(test_images_path, width, height, limit=-1)

    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    for (img_name, test_data) in get_images_for_tests(test_images_path, width, height, limit=-1):
        output = autoencoder.predict_proba(np.array([test_data]))
        pred = visualize(np.argmax(output[0], axis=1).reshape((height, width)), label_colours, nblbl)
        cv2.imwrite(os.path.join(prediction_output_path, img_name.split(".")[0]+".png"), pred)


def visualize(temp, label_colours, nblbl):
    black = np.zeros(temp.shape, np.uint8)
    r = black.copy()
    g = black.copy()
    b = black.copy()
    for l in range(0, nblbl):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = cv2.merge((r, g, b))
    return rgb


def save_history_graphs(history, title, path):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    outpath = os.path.join(path, "%s-%s.jpg" % (title, "acc"))
    plt.savefig(outpath)
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    outpath = os.path.join(path, "%s-%s.jpg" % (title, "loss"))
    plt.savefig(outpath)


def add_weights_entry(path, input_size, nb_epoch, batch_size, train_data_sz, val_data_sz, graph_path, magentize=True, normalize=True, data_augmentation=False, sample_per_epoch=0, nb_val_sample=0, comments=""):
    with open("./cnn/weights_table.txt", "a") as f:
        str_insize = "%dx%d" % input_size
        f.write("%s\t%s\t%d\t%d\t%d\t%d\t%r\t%r\t%r\t%d\t%d\t%s\t%s\n" % (path, str_insize, nb_epoch, batch_size, train_data_sz, val_data_sz, magentize, normalize, data_augmentation, sample_per_epoch, nb_val_sample, graph_path, comments))


def main(width, height, classes_weights):
    nblbl = 4
    nb_epoch = 100
    dataset_path = "./cnn/dataset/"
    test_images_path = "./cnn/test_images/"

    weigths_filepath = "./cnn/weigths/svf_%s.hdf5" % datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    #train_model(width, height, nblbl, dataset_path, weigths_filepath)
    #test_model(width, height, nblbl, test_images_path, weigths_filepath, "./predictions")
    train_model_generators(width, height, nblbl, classes_weights, dataset_path, weigths_filepath)
