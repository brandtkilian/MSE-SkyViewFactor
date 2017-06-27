import cv2
from tools.FileManager import FileManager
import os
import re
from sklearn.utils import shuffle
import random
import numpy as np
from enum import IntEnum
from tools.ImageTransform import ImageTransform
from core.ClassesEnum import Classes


class NormType(IntEnum):
    """The available equalization and normalization methods"""
    Equalize = 1
    EqualizeClahe = 2
    StdMean = 4
    Nothing = 8


class PossibleTransform(IntEnum):
    """The available images transformation for data augmentation"""
    Multiply = 0
    MultiplyPerChannels = 1
    GaussianNoise = 2
    Sharpen = 3
    AddSub = 5
    AddSubChannel = 7
    Invert = 8


class TransformDescriptor():
    """Class that encapsulate a transform on an image given a probability"""

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

    def __init__(self, src_directory, labels_directory, target_width, target_height, input_width=1440, input_height=1440, transforms=None, allow_transforms=False, rotate=False, lower_rotation_bound=0, higher_rotation_bound=180, norm_type=NormType.Equalize, magentize=True, batch_size=5, seed=1337, shuffled=True, yield_names=False, torify=True):
        self.src_directory = src_directory
        self.labels_directory = labels_directory
        self.width = target_width
        self.height = target_height
        self.input_width = input_width,
        self.input_height = input_height
        self.nblbl = 3 if torify else 4
        self.torify = torify
        self.angle = 360.

        if torify:
            radius = input_width / 2
            delta_r = float(radius) / target_height
            delta_angle = self.angle / target_width
            self.image_transform = ImageTransform(input_width, input_height, (input_width / 2, input_height / 2), radius, delta_angle, delta_r)

        self.input_width = input_width
        self.input_height = input_height

        self.lower_rotation_bound = lower_rotation_bound
        self.higher_rotation_bound = higher_rotation_bound
        self.norm_type = norm_type
        self.magentize = magentize
        self.allow_transform = allow_transforms
        self.rotate = rotate
        self.batch_size = batch_size
        self.seed = seed
        self.mask = self.get_void_mask(target_width, target_height)
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
        """Initialize a new images generation by generating rotation angles and shuffling images"""
        self.angles = [a for a in self.angles_generator(length)]
        if self.shuffled:
            self.img_files, self.lbl_files = shuffle(self.img_files, self.lbl_files)

    def angles_generator(self, length):
        """Generate random integer number between given angles boundaries"""
        for _ in range(length):
            yield random.randint(self.lower_rotation_bound, self.higher_rotation_bound)

    def image_generator(self, roll_axis=True):
        """Generate images following the transformations probabilities and format"""
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

                # if rotation allowed the rotate the image
                if self.rotate:
                    img = self.rotate_image(img, a)

                # it resampling allowed the resample image
                if self.torify:
                    img = self.image_transform.torify_image(img)
                else:
                    img = self.resize_if_needed(img)  # else resize image if needed

                # apply equalizations operations
                if self.norm_type & NormType.Equalize == NormType.Equalize:
                    img = self.normalize(img)
                elif self.norm_type & NormType.EqualizeClahe == NormType.EqualizeClahe:
                    img = self.normalize_clahe(img)

                # if transformations are allowed then shuffle the transform order
                if self.allow_transform:
                    random.shuffle(self.transforms_family)
                    # and apply the transformations on the image
                    for td in self.transforms_family:
                        img = td.call(img)

                # if the void class must be colored with magenta
                if self.magentize:
                    img = self.colorize_void(idx, color, img)  # fill void with magenta

                # once transformations beeing applied, normalize if any normalization is required
                if self.norm_type & NormType.StdMean == NormType.StdMean:
                    img = self.normalize_std(img)

                j += 1
                # rolling axis is for cnn else img is for visualization
                img = np.rollaxis(img, 2) if roll_axis else img
                # yield only image or name as well if required
                if self.yield_names:
                    yield img, name
                else:
                    yield img
                i += 1
            self.init_new_generation(length)  # ask for a new generation once every image has been yield

    def label_generator(self, binarized=True):
        """Generate formatted labels for the CNN followin the images transformations"""
        length = len(self.lbl_files)
        if len(self.angles) == 0:
            self.init_new_generation(length)

        idx = self.mask > 0
        while True:
            i = 0
            for a in self.angles:
                name = self.lbl_files[i % length]
                # if label should be binarized into a matrix (for cnn)
                # then load it as colore image else grayscale mode
                if binarized:
                    lbl = FileManager.LoadImage(name, self.labels_directory)
                else:
                    lbl = FileManager.LoadImage(name, self.labels_directory, cv2.IMREAD_GRAYSCALE)

                # if it should be rotated then rotate
                if self.rotate:
                    lbl = self.rotate_image(lbl, a, is_label=True)

                # if it should be resampled
                if self.torify:
                    lbl = self.image_transform.torify_image(lbl, interpolation=cv2.INTER_NEAREST)
                else:
                    lbl = self.resize_if_needed(lbl, is_label=True)

                # if void classe is magentized the fill the pixel with the void label value
                if self.magentize:
                    lbl[idx] = Classes.VOID

                # if it should be binarized (for cnn) then convert it into a binary matrix
                if binarized:
                    lbl = self.binarylab(lbl[:, :, 0], self.width, self.height, self.nblbl)
                    lbl = np.reshape(lbl, (self.width * self.height, self.nblbl))

                # yield the matrix or label image with or without the file name as required
                if self.yield_names:
                    yield lbl, name
                else:
                    yield lbl
                i += 1

    def image_batch_generator(self):
        """Uses the images generator method to yiel batches of images"""
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
        """Uses the labels generator method to yield batches of labels"""
        batch = []
        i = 0
        for lbl in self.label_generator(binarized):
            batch.append(lbl)
            i += 1
            if i % self.batch_size == 0:
                yield np.array(batch)
                batch = []
                i = 0

    def normalize_clahe(self, bgr):
        """Equalize the image histogram using clahe with a window of 1/30 image width"""
        b, g, r = cv2.split(bgr)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(int(self.width / 30), int(self.height / 30)))
        channels = (b, g, r)
        equalized = []
        for c in channels:
            equalized.append(clahe.apply(c))

        return cv2.merge(equalized)

    def resize_if_needed(self, img, is_label=False):
        """Resize an image if needed (not already the expected size). 
        If it is a label use nearest interpollation"""
        size = img.shape
        if size[1] != self.width or size[0] != self.height:
            mode = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
            resized = cv2.resize(img, (self.width, self.height), interpolation=mode)
            return resized

        return img

    @staticmethod
    def get_void_mask(width, height):
        """Get the mask that represents the void class for a given image.
        We make assumption that the circle containing information touch the borders and
        have a radius of width/2"""
        mask = np.zeros((height, width, 1), np.uint8)
        cv2.circle(mask, (width / 2, height / 2), width / 2, (255, 255, 255), -1)
        mask = cv2.bitwise_not(mask)
        return mask

    @staticmethod
    def colorize_void(idx, color, src_img):
        """Fill the void class with the given color"""
        b, g, r = cv2.split(src_img)
        b = b.reshape((b.shape[0], b.shape[1], 1))
        g = g.reshape((g.shape[0], g.shape[1], 1))
        r = r.reshape((r.shape[0], r.shape[1], 1))
        b[idx] = color[0]
        g[idx] = color[1]
        r[idx] = color[2]
        return cv2.merge((b, g, r))

    @staticmethod
    def normalize(bgr):
        """Equalize histogram of the bgr image channel per channel"""
        b, g, r = cv2.split(bgr)

        channels = (b, g, r)
        equalized = []
        for c in channels:
            equalized.append(cv2.equalizeHist(c))

        return cv2.merge(equalized)

    @staticmethod
    def normalize_std(rgb):
        """Normalize image pixels by subtracting mean and divinding by standard deviation"""
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
        """Convert an annotated label image into a binary matrix"""
        x = np.zeros([height, width, nblbl])
        for i in range(height):
            for j in range(width):
                x[i, j, labels[i][j]] = 1
        return x

    @staticmethod
    def rotate_image(image, angle, is_label=False):
        """Rotate an image by the given angle. If it is a label use neares interpollation."""
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        interpolation = cv2.INTER_CUBIC
        if is_label:
            interpolation = cv2.INTER_NEAREST
        res = cv2.warpAffine(image, M, (width, height), None, interpolation, cv2.BORDER_CONSTANT, (0, 0, 0))

        return res

    @staticmethod
    def add_or_sub_per_channel(image):
        """Add or sub a random value to each channel separatedly"""
        b, g, r = cv2.split(image)
        channels = [b, g, r]

        a_o_s = []
        for c in channels:
            a_o_s.append(ImageDataGenerator.add_or_sub(c))
        return cv2.merge(a_o_s)

    @staticmethod
    def add_or_sub(image):
        """Add or sub a random value to every pixels every channel (same value everywhere)"""
        bound = 20
        add_val = random.randint(0, bound) - int(bound/2)
        image = image.astype(np.int32)
        image += add_val

        return image.clip(0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_noise(image):
        """Apply a gaussian bluring with a kernel of random size [3, 7] (odd numbers only)"""
        kernel_size = random.randint(1, 3) * 2 + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    @staticmethod
    def multiply(image):
        """Mulitply all pixels (all channels) by a random value [0.9, 1.1]"""
        bound = 0.20
        image = image.astype(np.float64)
        factor = random.uniform(0.0, bound) - bound/2
        image *= (1 - factor)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def multiply_per_channel(image):
        """Multiply each channel of the bgr image by a random value [0.9, 1.1]"""
        b, g, r = cv2.split(image)
        chans = (b, g, r)
        multiplied = []
        for c in chans:
            multiplied.append(ImageDataGenerator.multiply(c))

        return cv2.merge(multiplied)

    @staticmethod
    def sharpen(image):
        """Sharpen edges of a given image by combining image and gaussian blurred image"""
        factor = random.uniform(0.2, 0.7)
        gaussian = cv2.GaussianBlur(image, (7, 7), 0)
        image = cv2.addWeighted(image, 1 + factor, gaussian, -factor, 0)
        return image

    @staticmethod
    def invert(image):
        """Returns the negative of the image"""
        return 255 - image

# Dictionnary a available transform to facilitate instantiation of image data generators
available_transforms = dict({PossibleTransform.Multiply: ImageDataGenerator.multiply,
                             PossibleTransform.AddSub: ImageDataGenerator.add_or_sub,
                             PossibleTransform.GaussianNoise: ImageDataGenerator.gaussian_noise,
                             PossibleTransform.MultiplyPerChannels: ImageDataGenerator.multiply_per_channel,
                             PossibleTransform.Sharpen: ImageDataGenerator.sharpen,
                             PossibleTransform.AddSubChannel: ImageDataGenerator.add_or_sub_per_channel,
                             PossibleTransform.Invert: ImageDataGenerator.invert,})
