import cv2
from tools.FileManager import FileManager
import os
import re
from sklearn.utils import shuffle
import random
import numpy as np
from enum import IntEnum


class NormType(IntEnum):
    Equalize = 0
    StdMean = 2
    Nothing = 1


class PossibleTransform(IntEnum):
    Multiply = 0
    MultiplyPerChannels = 1
    GaussianNoise = 2
    Sharpen = 3
    AddSub = 5
    Invert = 6


class TransformDescriptor():

    def __init__(self, callback, proba):
        self.proba = proba
        self.callback = callback
        print("Transform added %s with %1.2f probability of occurence" % (callback.__name__, proba))

    def call(self, image):
        lucky_or_not = random.uniform(0, 1) < self.proba
        if lucky_or_not:
            image = self.callback(image)
        return image


class ImageDataGenerator:

    def __init__(self, src_directory, labels_directory, width, height, nblbl, transforms=None, allow_transforms=False, rotate=False, lower_rotation_bound=0, higher_rotation_bound=180, norm_type=NormType.Equalize, magentize=True, batch_size=5, seed=1337, shuffled=True, yield_names=False):
        self.src_directory = src_directory
        self.labels_directory = labels_directory
        self.width = width
        self.height = height
        self.nblbl = nblbl
        self.lower_rotation_bound = lower_rotation_bound
        self.higher_rotation_bound = higher_rotation_bound
        self.norm_type = norm_type
        self.magentize = magentize
        self.allow_transform = allow_transforms
        self.rotate = rotate
        self.batch_size = batch_size
        self.seed = seed
        self.mask = self.get_void_mask(width, height)
        self.reg = r'\w+\.(jpg|jpeg|png)'
        self.img_files = sorted([f for f in os.listdir(self.src_directory) if re.match(self.reg, f.lower())])
        self.lbl_files = sorted([f for f in os.listdir(self.labels_directory) if re.match(self.reg, f.lower())])
        self.shuffled = shuffled
        self.yield_names = yield_names
        self.angles = []
        random.seed(self.seed)

        assert len(self.img_files) == len(self.lbl_files)

        self.transforms_family = []

        if transforms is not None:
            for t in transforms:
                try:
                    self.transforms_family.append(TransformDescriptor(available_transforms[t[0]], t[1]))
                except Exception as e:
                    print(e.message)

    def init_new_generation(self, length):
        self.angles = [a for a in self.angles_generator(length)]
        if self.shuffled:
            self.img_files, self.lbl_files = shuffle(self.img_files, self.lbl_files)

    def angles_generator(self, length):
        for _ in range(length):
            yield random.randint(self.lower_rotation_bound, self.higher_rotation_bound)

    def image_generator(self, roll_axis=True):
        color = (255, 0, 255)
        idx = self.mask > 0
        length = len(self.img_files)
        if len(self.angles) == 0:
            self.init_new_generation(length)
        j = 0
        while True:
            i = 0
            for a in self.angles:
                name = self.img_files[i % length]
                img = FileManager.LoadImage(name, self.src_directory)
                img = self.resize_if_needed(img)

                if self.norm_type == NormType.Equalize:
                    img = self.normalize(img)
                elif self.norm_type == NormType.StdMean:
                    img = self.normalize_std(img)

                if self.allow_transform:
                    random.shuffle(self.transforms_family)
                    for td in self.transforms_family:
                        img = td.call(img)
                if self.rotate:
                    img = self.rotate_image(img, a)

                if self.magentize:
                    img = self.colorize_void(idx, color, img)
                FileManager.SaveImage(img, "img%d.png" % j, "outputs/imgs_gen/")
                j += 1

                img = np.rollaxis(img, 2) if roll_axis else img
                if self.yield_names:
                    yield img, name
                else:
                    yield img
                i += 1
            self.init_new_generation(length)

    def label_generator(self, binarized=True):
        length = len(self.lbl_files)
        if len(self.angles) == 0:
            self.init_new_generation(length)
        while True:
            i = 0
            for a in self.angles:
                name = self.lbl_files[i % length]
                if binarized:
                    lbl = FileManager.LoadImage(name, self.labels_directory)
                else:
                    lbl = FileManager.LoadImage(name, self.labels_directory, cv2.IMREAD_GRAYSCALE)

                lbl = self.resize_if_needed(lbl, is_label=True)

                if self.rotate:
                    lbl = self.rotate_image(lbl, a, is_label=True)

                FileManager.SaveImage(lbl, "lbl%d.png" % i, "outputs/lbls_gen/")
                if binarized:
                    lbl = self.binarylab(lbl[:, :, 0], self.width, self.height, self.nblbl)
                    lbl = np.reshape(lbl, (self.width * self.height, self.nblbl))

                if self.yield_names:
                    yield lbl, name
                else:
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

    def label_batch_generator(self, binarized=True):
        batch = []
        i = 0
        for lbl in self.label_generator(binarized):
            batch.append(lbl)
            i += 1
            if i % self.batch_size == 0:
                yield np.array(batch)
                batch = []
                i = 0

    def normalize(self, rgb):
        b, g, r = cv2.split(rgb)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(int(self.width / 30), int(self.height / 30)))
        channels = (b, g, r)
        equalized = []
        for c in channels:
            equalized.append(clahe.apply(c))

        return cv2.merge(equalized)

    def resize_if_needed(self, img, is_label=False):
        size = img.shape
        if size[1] != self.width or size[0] != self.height:
            mode = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
            resized = cv2.resize(img, (self.width, self.height), interpolation=mode)
            return resized

        return img

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
    def normalize_std(rgb):
        b, g, r = cv2.split(rgb)

        channels = (b, g, r)
        normalized = []

        for c in channels:
            mean = cv2.mean(c)[0]
            std = np.std(c)

            normalized.append((c - mean) / std)

        return cv2.merge(normalized)


    @staticmethod
    def binarylab(labels, width, height, nblbl):
        x = np.zeros([height, width, nblbl])
        for i in range(height):
            for j in range(width):
                x[i, j, labels[i][j]] = 1
        return x

    @staticmethod
    def rotate_image(image, angle, is_label=False):
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        interpolation = cv2.INTER_CUBIC
        if is_label:
            interpolation = cv2.INTER_NEAREST
        res = cv2.warpAffine(image, M, (width, height), None, interpolation, cv2.BORDER_CONSTANT, (0, 0, 0))

        return res

    @staticmethod
    def add_or_sub(image):
        bound = 50
        add_val = random.randint(0, bound) - int(bound/2)
        image = image.astype(np.int32)
        image += add_val

        return image.clip(0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_noise(image):
        kernel_size = random.randint(1, 3) * 2 + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    @staticmethod
    def multiply(image):
        bound = 0.15
        image = image.astype(np.float64)
        factor = random.uniform(0.0, bound) - bound/2
        image *= (1 - factor)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def multiply_per_channel(image):
        b, g, r = cv2.split(image)
        chans = (b, g, r)
        multiplied = []
        for c in chans:
            multiplied.append(ImageDataGenerator.multiply(c))

        return cv2.merge(multiplied)

    @staticmethod
    def sharpen(image):
        factor = random.uniform(0.2, 0.7)
        gaussian = cv2.GaussianBlur(image, (7, 7), 0)
        image = cv2.addWeighted(image, 1 + factor, gaussian, -factor, 0)
        return image

    @staticmethod
    def invert(image):
        return 255 - image

available_transforms = dict({PossibleTransform.Multiply: ImageDataGenerator.multiply,
                    PossibleTransform.AddSub: ImageDataGenerator.add_or_sub,
                    PossibleTransform.GaussianNoise: ImageDataGenerator.gaussian_noise,
                    PossibleTransform.MultiplyPerChannels: ImageDataGenerator.multiply_per_channel,
                    PossibleTransform.Sharpen: ImageDataGenerator.sharpen,
                    PossibleTransform.Invert: ImageDataGenerator.invert,})
