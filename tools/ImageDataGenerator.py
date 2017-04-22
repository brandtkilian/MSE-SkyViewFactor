import cv2
from tools.FileManager import FileManager
import os
import re
from sklearn.utils import shuffle
import random
import numpy as np

class ImageDataGenerator:

    def __init__(self, src_directory, labels_directory, width, height, nblbl, lower_rotation_bound=0, higher_rotation_bound=180, normalize_img=True, magentize=True, rotate=False, batch_size=5, seed=1337, shuffled=True):
        self.src_directory = src_directory
        self.labels_directory = labels_directory
        self.width = width
        self.height = height
        self.nblbl = nblbl
        self.lower_rotation_bound = lower_rotation_bound
        self.higher_rotation_bound = higher_rotation_bound
        self.normalize_img = normalize_img
        self.magentize = magentize
        self.rotate = rotate
        self.batch_size = batch_size
        self.seed = seed
        self.mask = self.get_void_mask(width, height)
        self.reg = r'\w+\.(jpg|jpeg|png)'
        self.img_files = sorted([f for f in os.listdir(self.src_directory) if re.match(self.reg, f.lower())])
        self.lbl_files = sorted([f for f in os.listdir(self.labels_directory) if re.match(self.reg, f.lower())])

        assert len(self.img_files) == len(self.lbl_files)

        if shuffled:
            self.img_files, self.lbl_files = shuffle(self.img_files, self.lbl_files)

    def angles_generator(self):
        random.seed(self.seed)

        while True:
            yield random.randint(self.lower_rotation_bound, self.higher_rotation_bound)

    def image_generator(self):
        color = (255, 0, 255)
        idx = self.mask > 0
        i = 0
        length = len(self.img_files)
        for a in self.angles_generator():
            img = FileManager.LoadImage(self.img_files[i % length], self.src_directory)
            if self.normalize_img:
                img = self.normalize(img)
            if self.magentize:
                img = self.colorize_void(idx, color, img)
            if self.rotate:
                img = self.rotate_image(img, a)
            yield np.rollaxis(img, 2)
            i += 1

    def label_generator(self):
        i = 0
        length = len(self.lbl_files)
        for a in self.angles_generator():
            lbl = FileManager.LoadImage(self.lbl_files[i % length], self.labels_directory)
            if self.rotate:
                lbl = self.rotate_image(lbl, a, is_label=True)
            lbl = self.binarylab(lbl[:, :, 0], self.width, self.height, self.nblbl)
            lbl = np.reshape(lbl, (self.width * self.height, self.nblbl))
            yield lbl
            i += 1

    def image_batch_generator(self):
        batch = []
        i = 0
        for img in self.image_generator():
            batch.append(img)
            i += 1
            if i % self.batch_size == 0:
                yield np.array(batch)
                batch = []
                i = 0

    def label_batch_generator(self):
        batch = []
        i = 0
        for lbl in self.label_generator():
            batch.append(lbl)
            i += 1
            if i % self.batch_size == 0:
                yield np.array(batch)
                batch = []
                i = 0

    @staticmethod
    def get_void_mask(width, height):
        mask = np.zeros((height, width, 1), np.uint8)
        cv2.circle(mask, (width / 2, height / 2), width / 2, (255, 255, 255), -1)
        mask = cv2.bitwise_not(mask)
        return mask

    @staticmethod
    def colorize_void(idx, color, src_img):
        b, g, r = cv2.split(src_img)
        b = b.reshape((b.shape[0], b.shape[1], 1))
        g = g.reshape((g.shape[0], g.shape[1], 1))
        r = r.reshape((r.shape[0], r.shape[1], 1))
        b[idx] = color[0]
        g[idx] = color[1]
        r[idx] = color[2]
        return cv2.merge((b, g, r))

    @staticmethod
    def normalize(rgb):
        norm = np.zeros(rgb.shape, np.float32)

        b, g, r = cv2.split(rgb)

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

        return norm

    @staticmethod
    def binarylab(labels, width, height, nblbl):
        x = np.zeros([height, width, nblbl])
        for i in range(height):
            for j in range(width):
                x[i, j, labels[i][j]] = 1
        return x

    @staticmethod
    def rotate_image(image, angle, is_label=False):
        height, width, channels = image.shape
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        interpolation = cv2.INTER_CUBIC
        if is_label:
            interpolation = cv2.INTER_NEAREST
        res = cv2.warpAffine(image, M, (width, height), None, interpolation, cv2.BORDER_CONSTANT, (0, 0, 0))

        return res
