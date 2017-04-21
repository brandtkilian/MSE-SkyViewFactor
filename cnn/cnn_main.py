from __future__ import absolute_import
from __future__ import print_function
import os
from itertools import izip

from theano.gpuarray.opt import local_gpua_gemmbatch_output_merge

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

import tensorflow as tf
from .Model.ModelBuilding import create_model
import cv2
import numpy as np
from tools.FileManager import FileManager

def normalized(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def normalized_old(rgb):
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
    data_shape = width * height

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

    train_label = np.reshape(np.array(train_label), (len(train_data), data_shape, nblbl))
    return np.array(train_data), train_label, train_imgs, label_imgs


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


def get_images_for_tests(image_path, width, height, magentize=True, limit=10):
    img_names = sorted([f for f in os.listdir(image_path)])

    color = (255, 0, 255)
    mask = get_void_mask(width, height)
    idx = mask > 0

    length = len(img_names) if limit < 0 else limit

    for i in range(length):
        src_img = normalized(FileManager.LoadImage(img_names[i], image_path))
        if magentize:
            src_img = colorize_void(idx, color, src_img)

        yield img_names[i], np.rollaxis(src_img, 2)


def train_model(width, height, nblbl, dataset_path, weights_filepath):
    data_shape = width * height
    np.random.seed(1337)  # for reproducibility

    train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, nblbl)
    # train_label = np.reshape(train_label, (len(train_data), data_shape, nblbl))

    #class_weighting = [2.0, 4.61005688, 10.03329372, 5.45229053]  # dataset 100
    class_weighting = [1.99913108, 4.76866531,  9.54897594, 5.39499044]  # dataset 193

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        nb_epoch = 100
        batch_size = 1

        print(train_data.shape)
        print(train_label.shape)

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                                  class_weight=class_weighting, shuffle=True)

        autoencoder.save_weights(weights_filepath)


def train_model_generators(width, height, nblbl, classes_weights, dataset_path, weights_filepath):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        autoencoder = create_model(width, height, nblbl)

        nb_epoch = 100
        batch_size = 1
        img_gen_train, lbl_gen_train = create_data_augmentation_generators(dataset_path, "train", width, height, nblbl, batch_size)
        img_gen_valid, lbl_gen_valid = create_data_augmentation_generators(dataset_path, "tests", width, height, nblbl, batch_size)

        train_generator = izip(img_gen_train, lbl_gen_train)
        valid_generator = izip(img_gen_valid, lbl_gen_valid)

        s.run(tf.global_variables_initializer())
        history = autoencoder.fit_generator(train_generator, samples_per_epoch=300, nb_epoch=nb_epoch, verbose=1,
                                            validation_data=valid_generator, nb_val_samples=20, class_weight=[])

        autoencoder.save_weights(weights_filepath)

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


def create_data_augmentation_generators(dataset_path, root, width, height, nblbl, batch_size):
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    seed = 1
    # train_data, train_label, _, _ = prep_data(dataset_path, "train", width, height, nblbl)

    train_datagen_args = dict(
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=360,
        width_shift_range=0.,
        height_shift_range=0.,
        rescale=1,
        shear_range=0,
        zoom_range=0.,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=0.,
        fill_mode='constant',
        cval=0,)

    image_datagen = ImageDataGenerator(**train_datagen_args)
    labels_datagen = ImageDataGenerator(**train_datagen_args)

    image_generator = image_datagen.flow_from_directory(os.path.join(dataset_path, root, "src"), target_size=(width, height), classes=['.'], class_mode=None, seed=seed, batch_size=batch_size, color_mode='rgb')
    labels_generator = labels_datagen.flow_from_directory(os.path.join(dataset_path, root, "labels"), target_size=(width, height), classes=['.'], class_mode=None, seed=seed, batch_size=batch_size, color_mode='rgb')

    #train_generator = zip(image_generator, labels_generator)

    color = (255, 0, 255)
    mask = get_void_mask(height, width)
    idx = mask > 0

    def img_gen():
        for batch_img in image_generator:
            processed_batch = []
            for img in batch_img:
                img = img.astype(np.uint8)
                img = np.swapaxes(img, 0, 2)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = colorize_void(idx, color, img)
                img = normalized(img)
                img = np.rollaxis(img, 2)
                processed_batch.append(img)

            yield np.array(processed_batch)

    def lbl_gen():
        for batch_lbl in labels_generator:
            processed_batch = []
            for lbl in batch_lbl:
                lbl = lbl.astype(np.uint8)
                lbl = np.swapaxes(lbl, 0, 2)
                processed_batch.append(binarylab(lbl[:, :, 0], width, height, nblbl))

            yield np.reshape(np.array(processed_batch), (batch_size, width * height, nblbl))

    return img_gen(), lbl_gen()



def main(width, height, classes_weights):
    nblbl = 4
    nb_epoch = 100
    dataset_path = "./cnn/dataset/"
    test_images_path = "./cnn/test_images/"

    weigths_filepath = "./cnn/weigths_svf_ep%d_s%dx%d_magentavoid-data-augmented.hdf5" % (nb_epoch, width, height)

    #train_model(width, height, nblbl, dataset_path, weigths_filepath)
    #test_model(width, height, nblbl, test_images_path, weigths_filepath, "./predictions")
    train_model_generators(width, height, nblbl, classes_weights, dataset_path, weigths_filepath)
