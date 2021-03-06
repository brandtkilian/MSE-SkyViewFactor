import operator
import os
import re
import shutil
from itertools import izip

import cv2
import numpy as np
from sklearn.utils import shuffle

from core.ClassesEnum import Classes
from core.ImageDataGenerator import ImageDataGenerator, NormType, PossibleTransform
from tools.FileManager import FileManager
from tools.MaskCreator import MaskCreator


class DatasetManager:
    """Utility class to manage the dataset"""

    def __init__(self, valid_percentage, test_percentage, input_size, labels_path="images/labels", src_path="images/src", annot_output_path="outputs/annotated", dataset_output_path="outputs/dataset"):
        self.labels_path = labels_path
        self.src_path = src_path
        self.annot_output_path = annot_output_path
        self.dataset_output_path = dataset_output_path
        self.test_percentage = float(test_percentage)
        self.valid_percentage = float(valid_percentage)
        self.input_size = (1440, 1440)
        if isinstance(input_size, tuple):
            if len(input_size) == 2:
                self.input_size = input_size
            elif len(input_size) > 2:
                self.input_size = (self.input_size[0], self.input_size[1])

        self.mask = MaskCreator.create_circle_mask(input_size[0])

        self.reg = r'\w+\.(jpg|jpeg|png)'

        self.classes_weigth = dict({Classes.SKY: 0, Classes.BUILT: 0, Classes.VEGETATION: 0, Classes.VOID: 0})

    def remask_labels(self):
        """Apply a mask on every labels to mask every insignificant pixels (outside the fisheye cirlce)"""
        reg = r'\w+\.(jpg|jpeg|png)'
        files = [f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())]
        for f in files:
            imgSrc = FileManager.LoadImage(f, self.labels_path)
            if imgSrc.shape[0] != self.mask.shape[0] or imgSrc.shape[1] != self.mask.shape[1]:
                imgSrc = cv2.resize(imgSrc, (self.mask.shape[0], self.mask.shape[1]), interpolation=cv2.INTER_NEAREST)
            imgSrc = cv2.bitwise_and(imgSrc, imgSrc, None, self.mask)
            FileManager.SaveImage(imgSrc, f, self.labels_path)

    def check_for_labels_sanity(self, output_unsanity_masks_path="outputs/unsanityMask", output_sane_labels_path="outputs/labels"):
        """Check if every given labels only contains pure red, blue and green or black without collision else try to
        autocorrect the conflict or create a mask and notify user"""
        self.remask_labels()

        if not os.path.exists(output_unsanity_masks_path):
            os.makedirs(output_unsanity_masks_path)

        if not os.path.exists(output_sane_labels_path):
            os.makedirs(output_sane_labels_path)

        reg = r'\w+\.(jpg|jpeg|png)'
        files = [f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())]

        print "%d labels rgb images to proceed" % len(files)

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        nbUnsane = 0
        for f in files:
            imgSrc = FileManager.LoadImage(f, self.labels_path)
            b, g, r = cv2.split(imgSrc)

            conflicts = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_and(r, b),
                                       cv2.bitwise_and(r, g)), cv2.bitwise_and(b, r))

            nbConf = cv2.countNonZero(conflicts)

            if nbConf > 0:
                idx = conflicts > 0
                conflicts = conflicts.astype(np.uint8)
                conflicts = cv2.dilate(conflicts, ker)  # dilate to improve visibility
                gray = cv2.cvtColor(conflicts, cv2.COLOR_BAYER_BG2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                FileManager.SaveImage(thresh, f, output_unsanity_masks_path)
                imgSrc[idx] = (255, 255, 255)
                FileManager.SaveImage(imgSrc, f, self.labels_path)
                nbUnsane += 1
            else:
                shutil.copy(os.path.join(self.labels_path, FileManager.path_leaf(f)), os.path.join(output_sane_labels_path, FileManager.path_leaf(f)))

        print "%d labels images unsane detected, please check the unsanity masks in %s" % (nbUnsane, output_unsanity_masks_path)

        if nbUnsane == 0:
            shutil.rmtree(output_unsanity_masks_path)
            #shutil.rmtree(self.labels_path)
            self.labels_path = output_sane_labels_path
        return nbUnsane

    def create_annotated_images(self):
        """Create the annotated images, images with one integer value per classes
        from the rgb labels images"""
        if os.path.exists(self.dataset_output_path):
            return

        if not os.path.exists(self.annot_output_path):
            os.makedirs(self.annot_output_path)

        files = [f for f in os.listdir(self.labels_path) if re.match(self.reg, f.lower())]

        void_mask = cv2.bitwise_not(self.mask)
        nbVoid = cv2.countNonZero(void_mask)

        for f in files:
            imgSrc = FileManager.LoadImage(f, self.labels_path)
            b, g, r = cv2.split(imgSrc)

            annots = np.zeros(b.shape, np.uint8)

            ok = False
            i = 0
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            void_mask_ex = cv2.dilate(void_mask, ker)
            while not ok:
                bprime = cv2.dilate(b, ker)
                gprime = cv2.dilate(g, ker)
                rprime = cv2.dilate(r, ker)

                b = cv2.bitwise_or(cv2.bitwise_and(bprime, void_mask_ex), b)
                g = cv2.bitwise_or(cv2.bitwise_and(gprime, void_mask_ex), g)
                r = cv2.bitwise_or(cv2.bitwise_and(rprime, void_mask_ex), r)

                extended = cv2.bitwise_or(cv2.bitwise_or(b, g), r)
                extended = cv2.morphologyEx(extended, cv2.MORPH_CLOSE, ker)
                ok = abs(cv2.countNonZero(cv2.bitwise_and(extended, self.mask)) - cv2.countNonZero(self.mask)) < 5
                i += 1
                if i > 10:
                    print "No convergence found while filling holes between void and classes in image %s, please correct the image before continue" % f
                    return


            idxSky = b > 0
            idxVegetation = g > 0
            idxBuild = r > 0

            annots[idxSky] = Classes.SKY
            annots[idxVegetation] = Classes.VEGETATION
            annots[idxBuild] = Classes.BUILT
            annots[void_mask > 0] = Classes.VOID

            self.classes_weigth[Classes.SKY] += cv2.countNonZero(b)
            self.classes_weigth[Classes.VEGETATION] += cv2.countNonZero(g)
            self.classes_weigth[Classes.BUILT] += cv2.countNonZero(r)
            self.classes_weigth[Classes.VOID] += nbVoid

            FileManager.SaveImage(annots, f, self.annot_output_path)

        tot_pixels = sum(self.classes_weigth.values())
        self.classes_weigth = {k: 1.0/(v/float(tot_pixels)) if tot_pixels != 0 else 0 for k, v in self.classes_weigth.items()}

        print "Classes weigths ", self.classes_weigth
        self.labels_path = self.annot_output_path

    def create_final_dataset(self):
        """Create the final dataset by creating the directories hierarchy and 
        splitting the data following the given proportions"""
        if not os.path.exists(self.dataset_output_path):
            os.makedirs(self.dataset_output_path)
        else:
            print("A Dataset already exists, a new one won't be generated unless you remove it and rerun this script.")
            return

        reg = r'\w+\.(jpg|jpeg|png)'
        labels = sorted([f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())])
        src = sorted([f for f in os.listdir(self.src_path) if re.match(reg, f.lower())])

        assert len(src) == len(labels)

        valid_path = os.path.join(self.dataset_output_path, "valid")
        train_path = os.path.join(self.dataset_output_path, "train")
        test_path = os.path.join(self.dataset_output_path, "tests")

        shuffledSrc, shuffledLabels = shuffle(src, labels)

        test_path_labels = os.path.join(test_path, "labels")
        test_path_src = os.path.join(test_path, "src")

        train_path_labels = os.path.join(train_path, "labels")
        train_path_src = os.path.join(train_path, "src")

        valid_path_labels = os.path.join(valid_path, "labels")
        valid_path_src = os.path.join(valid_path, "src")

        if not os.path.exists(test_path_labels):
            os.makedirs(test_path_labels)
        if not os.path.exists(test_path_src):
            os.makedirs(test_path_src)
        if not os.path.exists(train_path_labels):
            os.makedirs(train_path_labels)
        if not os.path.exists(train_path_src):
            os.makedirs(train_path_src)
        if not os.path.exists(valid_path_src):
            os.makedirs(valid_path_src)
        if not os.path.exists(valid_path_labels):
            os.makedirs(valid_path_labels)

        boundaryValid = 0 if self.valid_percentage == 0 else int(len(labels) / 100.0 * self.valid_percentage)
        boundaryTests = 0 if self.test_percentage == 0 else int(len(labels) / 100.0 * self.test_percentage)
        print "%d images will be splitted and used for validation, %d for tests" % (boundaryValid, boundaryTests)

        trainSrc = shuffledSrc[boundaryTests + boundaryValid:]
        trainLabels = shuffledLabels[boundaryTests + boundaryValid:]

        validSrc = shuffledSrc[boundaryTests: boundaryValid + boundaryTests]
        validLabels = shuffledLabels[boundaryTests: boundaryValid + boundaryTests]

        testSrc = shuffledSrc[:boundaryTests]
        testLabels = shuffledLabels[:boundaryTests]

        print "Creating the training dataset"
        self.setup_dataset_split(train_path_labels, train_path_src, trainLabels, trainSrc)

        print "Creating the testing dataset"
        self.setup_dataset_split(test_path_labels, test_path_src, testLabels, testSrc)

        print "Creating the validation dataset"
        self.setup_dataset_split(valid_path_labels, valid_path_src, validLabels, validSrc)

    def setup_dataset_split(self, path_labels, path_src, labels, srcs):
        for i in range(len(srcs)):
            srcImg = FileManager.LoadImage(srcs[i], self.src_path)
            lblImg = FileManager.LoadImage(labels[i], self.labels_path, cv2.IMREAD_GRAYSCALE)

            FileManager.SaveImage(srcImg, srcs[i], path_src)
            FileManager.SaveImage(lblImg, labels[i], path_labels)

    def resize_images(self, input_dir, output_dir):
        """Resize a directory of images to the given size of desired dataset"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_names = sorted([f for f in os.listdir(input_dir)])

        for i in range(len(img_names)):
            img = FileManager.LoadImage(img_names[i], input_dir)
            resizedSrcImg = cv2.resize(img, self.input_size, interpolation=cv2.INTER_CUBIC)
            FileManager.SaveImage(resizedSrcImg, img_names[i], output_dir)

    def split_dataset_by_mostly_represented_class(self, input_src, input_annotated, mask, output_sky_most="outputs/sorted_dataset/sky/", output_veg_most="outputs/sorted_dataset/veg/",
                                                  output_build_most="outputs/sorted_dataset/build/", output_mixed="outputs/sorted_dataset/mixed/"):
        """Split the dataset into 4 sub-groups following the classes distribution.
        -Sky
        -Vegetation
        -Building
        -Mixed"""
        if not os.path.exists(output_sky_most):
            os.makedirs(output_sky_most)
        if not os.path.exists(output_veg_most):
            os.makedirs(output_veg_most)
        if not os.path.exists(output_build_most):
            os.makedirs(output_build_most)
        if not os.path.exists(output_mixed):
            os.makedirs(output_mixed)

        img_names = sorted([f for f in os.listdir(input_src) if re.match(self.reg, f.lower())])
        lbl_names = sorted([f for f in os.listdir(input_annotated) if re.match(self.reg, f.lower())])

        tot = cv2.countNonZero(mask)
        averages = [0., 0., 0.]
        for img_name, lbl_name in zip(img_names, lbl_names):
            lbl_src = FileManager.LoadImage(lbl_name, input_annotated, cv2.IMREAD_GRAYSCALE)
            img_src = FileManager.LoadImage(img_name, input_src)

            idx_sky = lbl_src == Classes.SKY
            idx_veg = lbl_src == Classes.VEGETATION
            idx_build = lbl_src == Classes.BUILT

            nb = idx_sky.sum() / float(tot)
            ng = idx_veg.sum() / float(tot)
            nr = idx_build.sum() / float(tot)

            outputs_tables = {0: output_sky_most, 1: output_veg_most, 2: output_build_most}

            percents = (nb, ng, nr)
            averages = map(operator.add, averages, percents)
            print averages[0]

            diffs = []
            for i in range(len(percents)):
                diffs.append(abs(percents[i-1] - percents[i]))

            max_diff = max(diffs)
            max_index, max_value = max(enumerate(percents), key=operator.itemgetter(1))

            if max_diff < 0.3:
                FileManager.SaveImage(lbl_src, lbl_name, os.path.join(output_mixed, "labels"))
                FileManager.SaveImage(img_src, img_name, os.path.join(output_mixed, "src"))
            else:
                FileManager.SaveImage(lbl_src, lbl_name, os.path.join(outputs_tables[max_index], "labels"))
                FileManager.SaveImage(img_src, img_name, os.path.join(outputs_tables[max_index], "src"))

        length = len(img_names)
        averages = map(lambda x: x / length, averages)
        return averages, length

    def create_synthetic_balanced_dataset_with_data_augmentation(self, input_src, input_annotated, mask, input_width, input_height,
                                                                 nblbl, output_sky_most="outputs/sorted_dataset/sky/",
                                                                 output_veg_most="outputs/sorted_dataset/veg/",
                                                                 output_build_most="outputs/sorted_dataset/build/",
                                                                 output_mixed="outputs/sorted_dataset/mixed/",
                                                                 output_path="outputs/synthetic_dataset/"):
        """Create a synthetic dataset that guaranty that every classes will be represented almost equally"""
        def compute_diffs(percents):
            diffs = []
            for i in range(len(percents)):
                diffs.append(abs(percents[i - 1] - percents[i]))
            return diffs

        averages, length = self.split_dataset_by_mostly_represented_class(input_src, input_annotated, mask,
                                                                          output_sky_most, output_veg_most,
                                                                          output_build_most, output_mixed)

        transforms = [(PossibleTransform.GaussianNoise, 0.1),
                      (PossibleTransform.Sharpen, 0.1),
                      (PossibleTransform.MultiplyPerChannels, 0.1),
                      (PossibleTransform.AddSub, 0.1),
                      (PossibleTransform.Multiply, 0.1), ]

        idg_sky = ImageDataGenerator(os.path.join(output_sky_most, "src"), os.path.join(output_sky_most, "labels"),
                                     input_width, input_height, nblbl,
                                     transforms=transforms, allow_transforms=True, norm_type=NormType.Nothing,
                                     magentize=False, rotate=True, shuffled=False)

        idg_veg = ImageDataGenerator(os.path.join(output_veg_most, "src"), os.path.join(output_veg_most, "labels"),
                                     input_width, input_height, nblbl,
                                     transforms=transforms, allow_transforms=True, norm_type=NormType.Nothing,
                                     magentize=False, rotate=True, shuffled=False)

        idg_build = ImageDataGenerator(os.path.join(output_build_most, "src"), os.path.join(output_build_most, "labels"),
                                     input_width, input_height, nblbl,
                                     transforms=transforms, allow_transforms=True, norm_type=NormType.Nothing,
                                     magentize=False, rotate=True, shuffled=False)

        sky_generator = izip(idg_sky.image_generator(roll_axis=False), idg_sky.label_generator(binarized=False))
        veg_generator = izip(idg_veg.image_generator(roll_axis=False), idg_veg.label_generator(binarized=False))
        build_generator = izip(idg_build.image_generator(roll_axis=False), idg_build.label_generator(binarized=False))

        generator_pool = {0: sky_generator, 1: veg_generator, 2: build_generator}

        tot = cv2.countNonZero(mask)
        max_diff = max(compute_diffs(averages))
        i = 0
        while max_diff > 0.05: # 5% difference
            min_index, _ = min(enumerate(averages), key=operator.itemgetter(1))
            gen = generator_pool[min_index]
            img, lbl = gen.next()
            FileManager.SaveImage(img, "%04d_augm.jpg" % i, os.path.join(output_path, "src"))
            FileManager.SaveImage(lbl, "%04d_augm.png" % i, os.path.join(output_path, "labels"))

            idx_sky = lbl == Classes.SKY
            idx_veg = lbl == Classes.VEGETATION
            idx_build = lbl == Classes.BUILT

            nb = idx_sky.sum() / float(tot)
            ng = idx_veg.sum() / float(tot)
            nr = idx_build.sum() / float(tot)

            percents = (nb, ng, nr)

            averages = map(lambda (ind, x): (x * length + percents[ind]) / (length + 1), enumerate(averages))
            length += 1
            i += 1
            print averages

            max_diff = max(compute_diffs(averages))
