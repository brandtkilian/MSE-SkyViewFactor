from __future__ import absolute_import
from __future__ import print_function

import os
from itertools import izip, product

from core.ImageDataGenerator import ImageDataGenerator, NormType, PossibleTransform
from core.BalancedImageDataGenerator import BalancedImageDataGenerator

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

import tensorflow as tf
from .model.ModelBuilding import create_model
from tools.FileManager import FileManager
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ntpath
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping
from core.SkyViewFactorCalculator import SkyViewFactorCalculator
from core.ClassesEnum import Classes
from tools.ImageTransform import ImageTransform

import random


def prep_data(base_path, from_path, width, height, torify, magentize=True):
    train_data = []
    train_label = []

    img_path = os.path.join(base_path, from_path, "src")
    lbl_path = os.path.join(base_path, from_path, "labels")

    idg = ImageDataGenerator(img_path, lbl_path, width, height, torify, allow_transforms=False, norm_type=NormType.Equalize, shuffled=False, magentize=magentize)

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


def get_images_generator_for_cnn(image_path, width, height, torify, magentize=True):
    idg = ImageDataGenerator(image_path, image_path, width, height, 4, magentize=magentize, allow_transforms=False, rotate=False, shuffled=False, torify=torify)
    return idg


def train_model(width, height, torify, dataset_path, weights_filepath, batch_size=6, nb_epoch=100):
    np.random.seed(1337)  # for reproducibility
    nblbl = Classes.nb_lbl(torify)
    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, torify)
    valid_data, valid_label, _, _ = prep_data(dataset_path, "valid", width, height, torify)

    #class_weighting = [2.0, 4.61005688, 10.03329372, 5.45229053]  # dataset 100
    class_weighting = [1.99913108, 4.76866531,  9.54897594, 5.39499044]  # dataset 193

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                                  class_weight=class_weighting, shuffle=True, validation_data=(valid_data, valid_label))

        autoencoder.save_weights(weights_filepath)

        comments = "Gaussian noise 3x3, no dropout, adadelta"

        graph_path = os.path.join("./cnn/weights/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(train_data),
                          len(valid_data), graph_path, data_augmentation=False, comments=comments, torify=torify)


def train_model_generators(width, height, torify, dataset_path, weights_filepath, nb_epoch=100,
                           batch_size=8, samples_per_epoch=200, samples_valid=-1, balanced=True):
    class_weighting = [1.999981, 4.88866531, 8.954169, 5.4417043]

    nblbl = Classes.nb_lbl(torify)
    img_path_train = os.path.join(dataset_path, "train", "src")
    lbl_path_train = os.path.join(dataset_path, "train", "labels")

    img_path_valid = os.path.join(dataset_path, "valid", "src")
    lbl_path_valid = os.path.join(dataset_path, "valid", "labels")

    transforms = [(PossibleTransform.GaussianNoise, 0.15),
                  (PossibleTransform.Sharpen, 0.15),
                  (PossibleTransform.MultiplyPerChannels, 0.15),
                  (PossibleTransform.AddSub, 0.15),
                  (PossibleTransform.Multiply, 0.15), ]

    if balanced:
        idg_train = BalancedImageDataGenerator(img_path_train, lbl_path_train, width, height, allow_transforms=True,
                                               rotate=True, transforms=transforms,
                                               lower_rotation_bound=0, higher_rotation_bound=360, magentize=True,
                                               norm_type=NormType.Equalize,
                                               batch_size=batch_size, seed=random.randint(1, 10e6), torify=torify)
    else:
        idg_train = ImageDataGenerator(img_path_train, lbl_path_train, width, height, allow_transforms=True, rotate=True, transforms=transforms,
                                       lower_rotation_bound=0, higher_rotation_bound=360, magentize=True, norm_type=NormType.Equalize,
                                       batch_size=batch_size, seed=random.randint(1, 10e6), torify=torify)

    idg_valid = ImageDataGenerator(img_path_valid, lbl_path_valid, width, height, magentize=True, norm_type=NormType.Equalize, rotate=False,
                                   batch_size=batch_size, seed=random.randint(1, 10e6), shuffled=False, torify=torify)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        train_generator = izip(idg_train.image_batch_generator(), idg_train.label_batch_generator())
        valid_generator = izip(idg_valid.image_batch_generator(), idg_valid.label_batch_generator())


        s.run(tf.global_variables_initializer())

        if samples_valid < 0:
            samples_valid = len(idg_valid.lbl_files)
        earlyStopping = EarlyStopping(monitor='loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
        history = autoencoder.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                            verbose=1, validation_data=valid_generator,
                                            nb_val_samples=samples_valid,
                                            class_weight=class_weighting, callbacks=[earlyStopping])

        autoencoder.save_weights(weights_filepath)

        comments = "equalizeCLAHE, adadelta, balanced %r" % balanced

        graph_path = os.path.join("./cnn/weights/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(idg_train.img_files),
                          len(idg_valid.img_files), graph_path, data_augmentation=True, sample_per_epoch=samples_per_epoch,
                          nb_val_sample=samples_valid, comments=comments, torify=torify)


def test_model(width, height, torify, test_images_path, weights_filepath, prediction_output_path, magentize=True):
    nblbl = Classes.nb_lbl(torify)
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]

    label_colours = np.array([Void, Sky, Building, Vegetation])
    idg = get_images_generator_for_cnn(test_images_path, width, height, torify, magentize=magentize)
    length = len(idg.img_files)

    i = 0
    for (img_name, test_data) in idg.image_generator():
        output = autoencoder.predict_proba(np.array([test_data]))
        reshaped_output = np.argmax(output[0], axis=1).reshape((height, width))
        pred = visualize(reshaped_output, label_colours, nblbl)

        if torify:
            pred = idg.image_transform.untorify_image(pred, interpolation=cv2.INTER_NEAREST)

        filename = img_name.split(".")[0]+".png"
        FileManager.SaveImage(pred, filename, prediction_output_path)

        i += 1
        if i >= length:
            break


def classify_images(images_path, weights_filepath, csv_output, save_outputs=False, classification_output_path="outputs/predictions", width=480, height=480, torify=True, magentize=True):
    nblbl = Classes.nb_lbl(torify)
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)
    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]
    label_colours = np.array([Void, Sky, Building, Vegetation])

    values = []

    idg = get_images_generator_for_cnn(images_path, width, height, torify, magentize=magentize)
    length = len(idg.img_files)

    i = 0
    with open(csv_output, "w") as f:
        f.write(",".join(["src_name", "SVF", "VVF", "BVF", "sky grav_center\n"]))
        for (img_name, test_data) in idg.image_generator():
            output = autoencoder.predict_proba(np.array([test_data]))
            reshaped_output = np.argmax(output[0], axis=1).reshape((height, width))
            pred = visualize(reshaped_output, label_colours, nblbl)

            if torify:
                pred = idg.image_transform.untorify_image(pred)

            factors = SkyViewFactorCalculator.compute_factor_bgr_labels(pred)
            b, g, r = cv2.split(pred)
            grav_center = SkyViewFactorCalculator.compute_sky_angle_estimation(b, (width/2, height/2), 0, width / 2)
            values.append(img_name)
            values.append("%.3f" % factors[0])
            values.append("%.3f" % factors[1])
            values.append("%.3f" % factors[2])
            values.append("%.3f" % grav_center)
            f.write(','.join(values) + "\n")
            values = []

            if save_outputs:
                filename = img_name.split(".")[0]+".png"
                FileManager.SaveImage(pred, filename, classification_output_path)

            i += 1
            if i >= length:
                break


def evaluate_model(width, height, nblbl, test_images_path, test_labels_path, weights_filepath, prediction_output_path):
    idg = ImageDataGenerator(test_images_path, test_labels_path, width, height, nblbl, norm_type=NormType.Equalize, magentize=True, rotate=False, shuffled=False)
    test_generator = izip(idg.image_generator(), idg.label_generator(binarized=False))

    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]

    label_colours = np.array([Void, Sky, Building, Vegetation])

    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    length = len(idg.img_files)

    i = 0
    true_labs = []
    pred_labs = []
    for src, lbl in test_generator:
        output = autoencoder.predict_proba(np.array([src]))
        max_output = np.argmax(output[0], axis=1)
        true_labs.append(lbl.reshape(width * height))
        pred_labs.append(max_output)
        pred = visualize(max_output.reshape(width, height), label_colours, nblbl)
        FileManager.SaveImage(pred, "pred%d.png" % i, prediction_output_path)
        i += 1
        if i == length:
            break

    true_labs = np.array(true_labs).ravel()
    pred_labs = np.array(pred_labs).ravel()
    cm = confusion_matrix(true_labs, pred_labs)
    target_names = ["Void", "Sky", "Veg", "Built"]
    plot_confusion_matrix(cm, target_names, False, output_filename=os.path.join(prediction_output_path, "confusion_matrix.jpg"))
    with open(os.path.join(prediction_output_path, "report.txt"), "w") as f:
        f.write(classification_report(true_labs, pred_labs, target_names=target_names))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output_filename="outputs/conf_mat.jpg"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_filename)


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


def add_weights_entry(path, input_size, nb_epoch, batch_size, train_data_sz, val_data_sz, graph_path, magentize=True, normalize=True, data_augmentation=False, sample_per_epoch=0, nb_val_sample=0, comments="", torify=True):
    with open("./cnn/weights_table.txt", "a") as f:
        str_insize = "%dx%d" % input_size
        f.write("%s\t%s\t%d\t%d\t%d\t%d\t%r\t%r\t%r\t%d\t%d\t%s\t%s\t%r\n" % (path, str_insize, nb_epoch, batch_size, train_data_sz, val_data_sz, magentize, normalize, data_augmentation, sample_per_epoch, nb_val_sample, graph_path, comments, torify))


def test_data_augmentation(dataset_path, width, height, nblbl):
    img_path = os.path.join(dataset_path, "train", "src")
    lbl_path = os.path.join(dataset_path, "train", "labels")

    transforms = [(PossibleTransform.GaussianNoise, 0.15),
                  (PossibleTransform.Sharpen, 0.15),
                  (PossibleTransform.MultiplyPerChannels, 0.15),
                  (PossibleTransform.AddSub, 0.15),
                  (PossibleTransform.Multiply, 0.15),]

    idg_test = ImageDataGenerator(img_path, lbl_path, width, height, nblbl, allow_transforms=True, rotate=True, transforms=transforms,
                                  lower_rotation_bound=10, higher_rotation_bound=350, magentize=True,
                                   norm_type=NormType.Equalize, seed=random.randint(1,1e6), shuffled=False)
    generator = izip(idg_test.image_generator(), idg_test.label_generator())

    i = 0
    for a, b in generator:
        if i > 200:
            break
        i += 1


def main(width, height, torify):
    nb_epoch = 100
    dataset_path = "./cnn/dataset/"
    test_images_path = "./cnn/test_images/"

    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #datestr = "2017-05-11_11:10:54"
    weigths_filepath = "./cnn/weights/svf_%s.hdf5" % datestr
    #train_model(width, height, nblbl, dataset_path, weigths_filepath)
    train_model_generators(width, height, torify, dataset_path, weigths_filepath, nb_epoch, balanced=True)
    evaluate_model(width, height, torify, "./cnn/dataset/tests/src", "./cnn/dataset/tests/labels", weigths_filepath, "./cnn/evaluations/predictions%s" % datestr)
    test_model(width, height, torify, test_images_path, weights_filepath=weigths_filepath, prediction_output_path="/home/brandtk/predictions%s" % datestr)
