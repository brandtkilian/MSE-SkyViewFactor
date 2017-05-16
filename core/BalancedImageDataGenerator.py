from core.ImageDataGenerator import ImageDataGenerator, NormType
from tools.FileManager import FileManager
from core.ClassesEnum import Classes
import cv2
import operator
from tools.MaskCreator import MaskCreator
from sklearn.utils import shuffle
import random
import numpy as np


class BalancedImageDataGenerator(ImageDataGenerator):

    def __init__(self, src_directory, labels_directory, width, height, nblbl, transforms=None, allow_transforms=False, rotate=False, lower_rotation_bound=0, higher_rotation_bound=180, norm_type=NormType.Equalize, magentize=True, batch_size=5, seed=1337, shuffled=True):
        ImageDataGenerator.__init__(self, src_directory, labels_directory, width, height=height, nblbl=nblbl, transforms=transforms, allow_transforms=allow_transforms,
                       rotate=rotate, lower_rotation_bound=lower_rotation_bound, higher_rotation_bound=higher_rotation_bound, norm_type=norm_type, magentize=magentize, batch_size=batch_size, seed=seed, shuffled=shuffled)

        self.veg_img = []
        self.built_img = []
        self.sky_img = []
        self.mixed_img = []

        self.veg_lbl = []
        self.built_lbl = []
        self.sky_lbl = []
        self.mixed_lbl = []

        self.map_list_img = {0: self.sky_img, 1: self.veg_img, 2: self.built_img}
        self.map_list_lbl = {0: self.sky_lbl, 1: self.veg_lbl, 2: self.built_lbl}

        self.mixed_probability = 0.5
        self.specific_class_probability = 0.5

        self.averages = [0, 0, 0]
        self.averages_classes = [0, 0, 0]
        self.classes_probabilities = [0.333, 0.333, 0.333]
        self.split_samples_into_lists()
        self.compute_probabilities()

        self.current_iteration_img = []
        self.current_iteration_lbl = []
        self.init_new_generation(len(self.img_files))

        self.occurences_dict = {}

    def compute_probabilities(self):
        self.mixed_probability = float(len(self.mixed_img)) / len(self.img_files)
        self.specific_class_probability = 1.0 - self.mixed_probability

        for i, a in enumerate(self.averages):
            self.classes_probabilities[i] = (1.0 / (a * self.averages_classes[i]))

        self.classes_probabilities = map(lambda x: x / sum(self.classes_probabilities), self.classes_probabilities)

        print "Probability of yielding a mixed image", self.mixed_probability
        print "Probability of yielding a specific class-most image", self.specific_class_probability
        print "Probabilities of thoses speicifics class-most images occurences sky %.2f, veg %.2f, built %.2f" % tuple(self.classes_probabilities)

    def split_samples_into_lists(self):

        def compute_diffs(percents):
            diffs = []
            for i in range(len(percents)):
                diffs.append(abs(percents[i - 1] - percents[i]))
            return diffs

        tot = cv2.countNonZero(MaskCreator.create_circle_maskwh(self.width, self.height))
        for img_name, lbl_name in zip(self.img_files, self.lbl_files):
            lbl_src = FileManager.LoadImage(lbl_name, self.labels_directory, cv2.IMREAD_GRAYSCALE)

            idx_sky = lbl_src == Classes.SKY
            idx_veg = lbl_src == Classes.VEGETATION
            idx_build = lbl_src == Classes.BUILT

            nb = idx_sky.sum() / float(tot)
            ng = idx_veg.sum() / float(tot)
            nr = idx_build.sum() / float(tot)

            percents = (nb, ng, nr)
            self.averages = map(operator.add, self.averages, percents)

            diffs = compute_diffs(percents)

            max_diff = max(diffs)
            max_index, max_value = max(enumerate(percents), key=operator.itemgetter(1))

            if max_diff < 0.3:
                self.mixed_img.append(img_name)
                self.mixed_lbl.append(lbl_name)
            else:
                self.map_list_img[max_index].append(img_name)
                self.map_list_lbl[max_index].append(lbl_name)
                self.averages_classes[max_index] += percents[max_index]

        length = len(self.img_files)
        self.averages = map(lambda x: x / length, self.averages)

        for i in range(len(self.map_list_img)):
            self.averages_classes[i] /= len(self.map_list_img[i])

        self.sort_lists()

    def sort_lists(self):
        self.sky_img = sorted(self.sky_img)
        self.veg_img = sorted(self.veg_img)
        self.built_img = sorted(self.built_img)
        self.sky_lbl = sorted(self.sky_lbl)
        self.veg_lbl = sorted(self.veg_lbl)
        self.built_lbl = sorted(self.built_lbl)
        self.mixed_img = sorted(self.mixed_img)
        self.mixed_lbl = sorted(self.mixed_lbl)

    def init_new_generation(self, length):
        self.angles = [a for a in self.angles_generator(length)]
        if self.shuffled:
            self.mixed_img, self.mixed_lbl = shuffle(self.mixed_img, self.mixed_lbl)
            self.sky_img, self.sky_lbl = shuffle(self.sky_img, self.sky_lbl)
            self.veg_img, self.veg_lbl = shuffle(self.veg_img, self.veg_lbl)
            self.built_img, self.built_lbl = shuffle(self.built_img, self.built_lbl)
        mixed_gen_img = self.build_generator(self.mixed_img)
        mixed_gen_lbl = self.build_generator(self.mixed_lbl)
        sky_gen_img = self.build_generator(self.sky_img)
        sky_gen_lbl = self.build_generator(self.sky_lbl)
        veg_gen_img = self.build_generator(self.veg_img)
        veg_gen_lbl = self.build_generator(self.veg_lbl)
        built_gen_img = self.build_generator(self.built_img)
        built_gen_lbl = self.build_generator(self.built_lbl)

        gens_table = {0: (sky_gen_img, sky_gen_lbl), 1: (veg_gen_img, veg_gen_lbl), 2: (built_gen_img, built_gen_lbl)}

        self.current_iteration_img = []
        self.current_iteration_lbl = []

        gen_img, gen_lbl = None, None

        for _ in range(length):
            mixed_or_not = random.uniform(0, 1)

            if self.mixed_probability - mixed_or_not < 0:
                r = random.uniform(0, 1)
                for i, p in enumerate(self.classes_probabilities):
                    r -= p
                    if r < 0:
                        gen_img, gen_lbl = gens_table[i]
                        break
            else:
                gen_img, gen_lbl = mixed_gen_img, mixed_gen_lbl

            self.current_iteration_img.append(gen_img.next())
            self.current_iteration_lbl.append(gen_lbl.next())

    def label_generator(self, binarized=True):
        length = len(self.current_iteration_lbl)
        while True:
            i = 0
            for a in self.angles:
                print a
                if binarized:
                    lbl = FileManager.LoadImage(self.current_iteration_lbl[i % length], self.labels_directory)
                else:
                    lbl = FileManager.LoadImage(self.current_iteration_lbl[i % length], self.labels_directory, cv2.IMREAD_GRAYSCALE)

                lbl = self.resize_if_needed(lbl, is_label=True)

                if self.rotate:
                    lbl = self.rotate_image(lbl, a, is_label=True)
                if binarized:
                    lbl = self.binarylab(lbl[:, :, 0], self.width, self.height, self.nblbl)
                    lbl = np.reshape(lbl, (self.width * self.height, self.nblbl))
                yield lbl
                i += 1
                self.occurences_dict[self.current_iteration_lbl[i % length]] = self.occurences_dict.get(self.current_iteration_lbl[i % length], 0) + 1

    def image_generator(self, roll_axis=True):
        color = (255, 0, 255)
        idx = self.mask > 0
        length = len(self.current_iteration_img)
        j = 0
        while True:
            i = 0
            for a in self.angles:
                print a
                img = FileManager.LoadImage(self.current_iteration_img[i % length], self.src_directory)
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
                j += 1

                yield np.rollaxis(img, 2) if roll_axis else img
                i += 1
            self.init_new_generation(length)

    def build_generator(self, list):
        def gen():
            counter = 0
            length = len(list)
            while True:
                yield list[counter]
                counter = (counter + 1) % length
        return gen()


