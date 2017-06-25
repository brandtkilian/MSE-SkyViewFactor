from __future__ import absolute_import
from __future__ import print_function

import os
from itertools import izip, product

from core.ImageDataGenerator import ImageDataGenerator, NormType, PossibleTransform
from core.BalancedImageDataGenerator import BalancedImageDataGenerator

# global variables
os.environ['KERAS_BACKEND'] = 'tensorflow'  # using tensorflow backend
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # visible GPU device ID
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"  # path to cuda libraries

import tensorflow as tf
from .model.ModelBuilding import create_model
from tools.FileManager import FileManager
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ntpath
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from core.SkyViewFactorCalculator import SkyViewFactorCalculator
from core.ClassesEnum import Classes
from imgproc_main import svf_graph_and_mse

import random


def prep_data(base_path, from_path, width, height, norm_type=NormType.Equalize,
              torify=False, magentize=True):
    """Prepare data for training"""
    train_data = []
    train_label = []

    img_path = os.path.join(base_path, from_path, "src")
    lbl_path = os.path.join(base_path, from_path, "labels")

    idg = ImageDataGenerator(img_path, lbl_path, width, height, allow_transforms=False, norm_type=norm_type,
                             shuffled=False, magentize=magentize, torify=torify)

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


def get_images_generator_for_cnn(image_path, width, height, norm_type=NormType.Equalize, magentize=True, torify=False, yield_names=False):
    idg = ImageDataGenerator(image_path, image_path, width, height, magentize=magentize, norm_type=norm_type,
                             allow_transforms=False, rotate=False, shuffled=False, torify=torify,
                             yield_names=yield_names)
    """Instantiate a basic data augmentor with no transforms and no rotation used for training or testing without augmentation
    The purpose of this is to apply preprocessing steps like resizing or normalizations"""
    return idg


def train_model(width, height, dataset_path, weights_filepath, batch_size=6, nb_epoch=100, class_weights=None,
                magentize=True, norm_type=NormType.Equalize, torify=False, early_stopping=False):
    """Train the model using no data augmentation"""
    np.random.seed(1337)  # for reproducibility
    nblbl = Classes.nb_lbl(torify)
    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, torify=torify,
                                              magentize=magentize, norm_type=norm_type)
    valid_data, valid_label, _, _ = prep_data(dataset_path, "valid", width, height, torify=torify,
                                              magentize=magentize, norm_type=norm_type)

    class_weights = [1.0 for _ in range(nblbl)] if class_weights is None else class_weights[:nblbl]


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        min_delta = 1e-3
        patience = 20
        monitor = 'loss'
        mode = 'auto'
        earlyStopping = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode)
        best_weight_filepath = weights_filepath + ".best"
        print(best_weight_filepath)
        model_checkpoint = ModelCheckpoint(best_weight_filepath, "val_loss", save_best_only=True, mode="min",
                                           save_weights_only=True)
        callbacks = [model_checkpoint]
        if early_stopping:
            callbacks.append(earlyStopping)

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                                  class_weight=class_weights, shuffle=True, validation_data=(valid_data, valid_label),
                                  callbacks=callbacks)

        autoencoder.save_weights(weights_filepath)

        comments = "trained without data augmentation, early_stopping %r " \
                   "(monitor=%s, min_delta=%f, patience=%d, mode=%s)" % (early_stopping, monitor,
                                                                         min_delta, patience, mode)

        graph_path = os.path.join("./cnn/weights/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(train_data),
                          len(valid_data), weights_filepath, graph_path, norm_type=norm_type, data_augmentation=False,
                          comments=comments, torify=torify)


def train_model_generators(width, height, dataset_path, weights_filepath, nb_epoch=100,
                           batch_size=6, class_weights=None, samples_per_epoch=200, samples_valid=-1, balanced=True,
                           norm_type=NormType.Equalize, magentize=True, torify=False, early_stopping=False):
    """Train a model using data augmentation"""

    nblbl = Classes.nb_lbl(torify)
    img_path_train = os.path.join(dataset_path, "train", "src")
    lbl_path_train = os.path.join(dataset_path, "train", "labels")

    img_path_valid = os.path.join(dataset_path, "valid", "src")
    lbl_path_valid = os.path.join(dataset_path, "valid", "labels")

    probabilities = 0.10
    transforms = [(PossibleTransform.GaussianNoise, probabilities),
                  (PossibleTransform.Sharpen, probabilities),
                  (PossibleTransform.MultiplyPerChannels, probabilities),
                  (PossibleTransform.AddSub, probabilities),
                  (PossibleTransform.Multiply, probabilities),
                  (PossibleTransform.AddSubChannel, probabilities)]
    n_transforms = len(transforms)

    if balanced:
        class_weights = [1.0 for _ in range(nblbl)]
        idg_train = BalancedImageDataGenerator(img_path_train, lbl_path_train, width, height, allow_transforms=True,
                                               rotate=True, transforms=transforms,
                                               lower_rotation_bound=0, higher_rotation_bound=360, magentize=magentize,
                                               norm_type=norm_type,
                                               batch_size=batch_size, seed=random.randint(1, 10e6), torify=torify)
    else:
        class_weights = [1.0 for _ in range(nblbl)] if class_weights is None else class_weights[:nblbl]
        idg_train = ImageDataGenerator(img_path_train, lbl_path_train, width, height, allow_transforms=True,
                                       rotate=True, transforms=transforms,
                                       lower_rotation_bound=0, higher_rotation_bound=360, magentize=magentize,
                                       norm_type=norm_type,
                                       batch_size=batch_size, seed=random.randint(1, 10e6), torify=torify)

    idg_valid = ImageDataGenerator(img_path_valid, lbl_path_valid, width, height, magentize=magentize, norm_type=norm_type, rotate=False,
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
        min_delta=1e-2
        patience=15
        monitor='loss'
        mode='auto'
        earlyStopping = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=1, mode=mode)
        best_weight_filepath = weights_filepath + ".best"
        model_checkpoint = ModelCheckpoint(best_weight_filepath, "val_loss", save_best_only=True, mode="min",
                                           save_weights_only=True)
        callbacks = [model_checkpoint]
        if early_stopping:
            callbacks.append(earlyStopping)
        history = autoencoder.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                            verbose=1, validation_data=valid_generator,
                                            nb_val_samples=samples_valid,
                                            class_weight=class_weights, callbacks=callbacks)

        autoencoder.save_weights(weights_filepath)

        comments = "trained with data augmentation, balanced generator %r, proba transform %f, nb transforms %d, early_stopping %r " \
                   "(monitor=%s, min_delta=%f, patience=%d, mode=%s)" % (balanced, probabilities, n_transforms,
                                                                         early_stopping, monitor,
                                                                         min_delta, patience, mode)

        graph_path = os.path.join("./cnn/weights/graphs", ntpath.basename(weights_filepath).split(".")[0])
        save_history_graphs(history, "model", graph_path)
        add_weights_entry(weights_filepath, (width, height), nb_epoch, batch_size, len(idg_train.img_files),
                          len(idg_valid.img_files), weights_filepath, graph_path, data_augmentation=True,
                          sample_per_epoch=samples_per_epoch, norm_type=norm_type,
                          nb_val_sample=samples_valid, comments=comments, torify=torify)


def test_model(width, height, torify, test_images_path, weights_filepath, prediction_output_path,
               norm_type=NormType.Equalize, magentize=True):
    """Test a model by classifying a whole images folder"""
    nblbl = Classes.nb_lbl(torify)
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 255, 0]
    Vegetation = [0, 0, 255]
    Void = [0, 0, 0]

    label_colours = np.array([Sky, Building, Vegetation, Void])
    idg = get_images_generator_for_cnn(test_images_path, width, height, norm_type=norm_type, magentize=magentize, torify=torify)
    idg.yield_names = True
    length = len(idg.img_files)

    i = 0
    for (test_data, img_name) in idg.image_generator():
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


def classify_images(images_path, weights_filepath, csv_output, save_outputs=False, save_overlay=False, classification_output_path="outputs/predictions", width=480, height=480,
                    norm_type=NormType.EqualizeClahe, magentize=True, torify=False, gravity_angle=False):
    """Method used by the production main that classify a whole folder of images and computing the differents view factors
    Store the data generated into the specified folder and csv file"""
    nblbl = Classes.nb_lbl(torify)
    # create the model
    autoencoder = create_model(width, height, nblbl)
    # load the given weights
    autoencoder.load_weights(weights_filepath)
    # colors for the predictions images visualization
    Sky = [255, 0, 0]
    Building = [0, 0, 255]
    Vegetation = [0, 255, 0]
    Void = [0, 0, 0]
    label_colours = np.array([Sky, Vegetation, Building, Void])

    values = []
    # create the output directory if not exists
    directory = os.path.dirname(csv_output)
    if not os.path.exists(directory):
        os.makedirs(directory)

    overlay_path = os.path.join(classification_output_path, "overlays")

    # headers of column in the csv output file
    headers = ["src_name", "SVF", "VVF", "BVF", "sky grav_center\n"] if gravity_angle else ["src_name", "SVF", "VVF", "BVF\n"]

    # create the csv output file and write headers
    with open(csv_output, "w+") as f:
        f.write(",".join(headers))

    # get the image generator that will yield images from the directory to classify
    idg = get_images_generator_for_cnn(images_path, width, height,
                                       norm_type=norm_type, magentize=magentize,
                                       torify=torify, yield_names=True)
    length = len(idg.img_files)

    i = 0
    with open(csv_output, "w") as f:
        f.write(",".join(["src_name", "SVF", "VVF", "BVF", "sky", "grav_center_angle\n"]))
        for (img, img_name) in idg.image_generator():
            output = autoencoder.predict_proba(np.array([img]))
            reshaped_output = np.argmax(output[0], axis=1).reshape((height, width))
            pred = visualize(reshaped_output, label_colours, nblbl)

            if torify:
                pred = idg.image_transform.untorify_image(pred)

            center = (pred.shape[1] / 2, pred.shape[0] / 2)
            radius = pred.shape[1] / 2
            factors = SkyViewFactorCalculator.compute_factor_bgr_labels(pred, center=center, radius=radius)
            values.append(img_name)
            values.append("%.5f" % factors[0])
            values.append("%.5f" % factors[1])
            values.append("%.5f" % factors[2])

            if gravity_angle:
                b, g, r = cv2.split(pred)
                grav_center = SkyViewFactorCalculator.compute_sky_angle_estimation(b, center=center, radius_low=0,
                                                                                   radius_top=radius,
                                                                                   center_factor=center,
                                                                                   radius_factor=radius,
                                                                                   sky_view_factor=factors[0])
                rad_pixels = grav_center * (radius / 90)
                cv2.circle(pred, center, int(rad_pixels), (255, 255, 255), 1)
                values.append("%.5f" % grav_center)

            f.write(','.join(values) + "\n")
            values = []

            if save_outputs:
                filename = img_name.split(".")[0]+".png"
                FileManager.SaveImage(pred, filename, classification_output_path)

            if save_overlay:
                overlayed = create_overlay_image(img_name, pred, images_path)
                FileManager.SaveImage(overlayed, img_name, overlay_path)

            i += 1
            if i >= length:
                break


def evaluate_model(width, height, test_images_path, test_labels_path,
                   weights_filepath, prediction_output_path, norm_type=NormType.Equalize, magentize=True, torify=False):
    """Evaluate a trained model by computing classification report and confusion matrix
    Store the predictions as well"""
    # using the Image Data Generator utility class with not shufflind, no rotation, no transformation
    idg = ImageDataGenerator(test_images_path, test_labels_path, width, height,
                             norm_type=norm_type, magentize=magentize, rotate=False,
                             shuffled=False, yield_names=True, torify=torify)

    test_generator = izip(idg.image_generator(), idg.label_generator(binarized=False))

    nblbl = Classes.nb_lbl(torify)
    autoencoder = create_model(width, height, nblbl)
    autoencoder.load_weights(weights_filepath)

    Sky = [255, 0, 0]
    Building = [0, 0, 255]
    Vegetation = [0, 255, 0]
    Void = [0, 0, 0]

    label_colours = np.array([Sky, Vegetation, Building, Void])
    # create output directory if not exists
    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    length = len(idg.img_files)

    overlays_path = os.path.join(prediction_output_path, "overlays")
    i = 0
    true_labs = []
    pred_labs = []
    # iterate over tests images
    for src_info, lbl_info in test_generator:
        src = src_info[0]
        lbl = lbl_info[0]
        name = src_info[1]
        # use the model to predict classes
        output = autoencoder.predict_proba(np.array([src]))
        # keep max probabilities
        max_output = np.argmax(output[0], axis=1)
        true_labs.append(lbl.reshape(width * height))
        pred_labs.append(max_output)
        # create the visualization image from prediction (blue, green, red)
        pred = visualize(max_output.reshape(height, width), label_colours, nblbl)
        if torify:
            pred = idg.image_transform.untorify_image(pred, cv2.INTER_NEAREST)
        FileManager.SaveImage(pred, name.split(".")[0]+".png", prediction_output_path)
        overlayed = create_overlay_image(name, pred, test_images_path)
        FileManager.SaveImage(overlayed, name, overlays_path)

        i += 1
        # as we use the generator, exit the loop once every image has been used once
        if i == length:
            break

    true_labs = np.array(true_labs).ravel()
    pred_labs = np.array(pred_labs).ravel()
    # compute confusion matrix using ground truth and predictions
    cm = confusion_matrix(true_labs, pred_labs)
    target_names = ["Sky", "Veg", "Built", "Void"]
    if torify:
        target_names = target_names[:-1]
    # plot the matrix into an image and save it in the output folder
    plot_confusion_matrix(cm, target_names, True, output_filename=os.path.join(prediction_output_path, "confusion_matrix.jpg"))
    # Generate the classification report and write it on disk
    with open(os.path.join(prediction_output_path, "report.txt"), "w") as f:
        f.write(classification_report(true_labs, pred_labs, target_names=target_names, digits=5))
    # finally generate the MSE graphs
    svf_graph_and_mse(test_labels_path, prediction_output_path, prediction_output_path)


def create_overlay_image(name, pred, src_images_path):
    """Create an overlayed image by combining source and prediction"""
    overlay_src = FileManager.LoadImage(name, src_images_path)
    overlay_src = cv2.resize(overlay_src, pred.shape[:2])
    overlayed = cv2.addWeighted(overlay_src.astype(np.uint8), 0.8, pred.astype(np.uint8), 0.2, 0)
    return overlayed


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
        if cm[j, i] < 1e-3:
            format_nb = "%.3E"
        else:
            format_nb = "%.3f"
        plt.text(j, i, format_nb % cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_filename)


def visualize(temp, label_colours, nblbl):
    """Convert the output of the classifier into a BGR image for predictions visualization"""
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
    """Plot the loss/accuracy graph evolution over epochs and save the figure in the specified path"""
    if not os.path.exists(path):
        os.makedirs(path)
    plt.cla()
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
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


def add_weights_entry(path, input_size, nb_epoch, batch_size, train_data_sz, val_data_sz, weights_filepath, graph_path, magentize=True, norm_type=NormType.Equalize, data_augmentation=False, sample_per_epoch=0, nb_val_sample=0, comments="", torify=True):
    """Add the entry of a just finished training weights in the weights table file with used parameters and comments"""
    with open("./cnn/weights_table.txt", "a") as f:
        str_insize = "%dx%d" % input_size
        cmd = "python main.py -i path/to/inputs -o /path/to/output --width %d --height %d -w %s --csv-file path/to/csv -n %d" % (input_size[0], input_size[1], os.path.abspath(weights_filepath), int(norm_type))
        cmd += " -t" if torify else ""
        cmd += " -m" if magentize else ""
        f.write("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%r\t%d\t%d\t%s\t%s\t%r\t%s\n" % (path, str_insize, nb_epoch, batch_size, train_data_sz, val_data_sz, magentize, norm_type, data_augmentation, sample_per_epoch, nb_val_sample, graph_path, comments, torify, cmd))


def test_data_augmentation(dataset_path, width, height, nblbl):
    """Just generate augmented images for tests purpose"""
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


def main(class_weigths):
    """Main"""
    nb_epoch = 200
    batch_size = 6
    norm_type = NormType.EqualizeClahe + NormType.SPHcl
    magentize = True
    width = 480
    height = 480
    torify = not magentize


    dataset_path = "./cnn/dataset/"
    test_images_path = "/home/brandtk/Desktop/SVF/outputs_NE"

    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    datestr = "2017-06-08_12:07:31"
    weigths_filepath = "./cnn/weights/svf_%s.hdf5" % datestr

    #train_model(width, height, dataset_path, weigths_filepath, nb_epoch=nb_epoch, batch_size=batch_size,
    #            class_weights=class_weigths, norm_type=norm_type, magentize=magentize, torify=torify,
    #            early_stopping=False)
    #evaluate_model(width, height, "./cnn/dataset/tests/src", "./cnn/dataset/tests/labels",
    #               weigths_filepath+".best", "./cnn/evaluations/predictions%s" % datestr,
    #               norm_type=norm_type, magentize=magentize, torify=torify)

    #train_model_generators(width, height, dataset_path, weigths_filepath, nb_epoch=nb_epoch, batch_size=batch_size,
    #                       class_weights=class_weigths, norm_type=norm_type, balanced=True, magentize=magentize,
    #                       torify=torify, early_stopping=False)
    evaluate_model(width, height, "./cnn/dataset/tests/src", "./cnn/dataset/tests/labels",
                   weigths_filepath + ".best", "./cnn/evaluations/predictions%s" % datestr,
                   norm_type=norm_type, magentize=magentize, torify=torify)
    #test_model(width, height, torify, test_images_path, weights_filepath=weigths_filepath, prediction_output_path="/home/brandtk/predictions%s" % datestr)
