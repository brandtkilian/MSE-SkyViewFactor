import cv2
import os
import re
from tools.FileManager import FileManager
import shutil
import numpy as np
from core.ClassesEnum import Classes
from sklearn.utils import shuffle


class DatasetManager:

    def __init__(self, mask, test_percentage, target_size, labels_path="images/labels", src_path="images/src", annot_output_path="outputs/annoted", dataset_output_path="outputs/dataset"):
        self.mask = mask
        self.labels_path = labels_path
        self.src_path = src_path
        self.annot_output_path = annot_output_path
        self.dataset_output_path = dataset_output_path
        self.test_percentage = float(test_percentage)
        self.targetSize = (360, 360)
        if isinstance(target_size, tuple):
            if len(target_size) == 2:
                self.targetSize = target_size
            elif len(target_size) > 2:
                self.targetSize = (self.targetSize[0], self.targetSize[1])

        self.classes_weigth = dict({Classes.SKY: 0, Classes.BUILT: 0, Classes.VEGETATION: 0, Classes.VOID: 0})

    def remaskLabels(self):
        reg = r'\w+\.(jpg|jpeg|png)'
        files = [f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())]
        for f in files:
            imgSrc = FileManager.LoadImage(f, self.labels_path)
            imgSrc = cv2.bitwise_and(imgSrc, imgSrc, None, self.mask)
            FileManager.SaveImage(imgSrc, f, self.labels_path)

    def fillVoidClassInSource(self, src, idx, color):
        assert len(color) == 3

        b, g, r = cv2.split(src)
        b = b.reshape((b.shape[0], b.shape[1], 1))
        g = g.reshape((g.shape[0], g.shape[1], 1))
        r = r.reshape((r.shape[0], r.shape[1], 1))
        b[idx] = color[0]
        g[idx] = color[1]
        r[idx] = color[2]
        return cv2.merge((b, g, r))

    def checkForLabelsSanity(self, output_unsanity_masks_path="outputs/unsanityMask", output_sane_labels_path="outputs/labels"):

        self.remaskLabels()

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
                shutil.move(os.path.join(self.labels_path, FileManager.path_leaf(f)), os.path.join(output_sane_labels_path, FileManager.path_leaf(f)))

        print "%d labels images unsane detected, please check the unsanity masks in %s" % (nbUnsane, output_unsanity_masks_path)

        if nbUnsane == 0:
            shutil.rmtree(output_unsanity_masks_path)
            #shutil.rmtree(self.labels_path)
            self.labels_path = output_sane_labels_path
        return nbUnsane

    def createAnotedImages(self):
        if not os.path.exists(self.annot_output_path):
            os.makedirs(self.annot_output_path)

        reg = r'\w+\.(jpg|jpeg|png)'
        files = [f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())]

        nbVoid = cv2.countNonZero(self.mask)

        for f in files:
            imgSrc = FileManager.LoadImage(f, self.labels_path)
            b, g, r = cv2.split(imgSrc)

            annots = np.zeros(b.shape, np.uint8)

            idxSky = b > 0
            idxVegetation = g > 0
            idxBuild = r > 0

            annots[idxSky] = Classes.SKY
            annots[idxVegetation] = Classes.VEGETATION
            annots[idxBuild] = Classes.BUILT

            self.classes_weigth[Classes.SKY] += cv2.countNonZero(b)
            self.classes_weigth[Classes.VEGETATION] += cv2.countNonZero(g)
            self.classes_weigth[Classes.BUILT] += cv2.countNonZero(r)
            self.classes_weigth[Classes.VOID] += nbVoid

            FileManager.SaveImage(annots, f, self.annot_output_path)

        tot_pixels = sum(self.classes_weigth.values())
        self.classes_weigth = {k: 1/(v/float(tot_pixels)) for k, v in self.classes_weigth.items()}

        print "Classes weigths ", self.classes_weigth
        self.labels_path = self.annot_output_path

    def createFinalDataset(self):
        if not os.path.exists(self.dataset_output_path):
            os.makedirs(self.dataset_output_path)

        reg = r'\w+\.(jpg|jpeg|png)'
        labels = sorted([f for f in os.listdir(self.labels_path) if re.match(reg, f.lower())])
        src = sorted([f for f in os.listdir(self.src_path) if re.match(reg, f.lower())])

        assert len(src) == len(labels)

        test_path = os.path.join(self.dataset_output_path, "tests")
        train_path = os.path.join(self.dataset_output_path, "train")

        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        shuffledSrc, shuffledLabels = shuffle(src, labels)

        test_path_labels = os.path.join(test_path, "labels")
        test_path_src = os.path.join(test_path, "src")

        train_path_labels = os.path.join(train_path, "labels")
        train_path_src = os.path.join(train_path, "src")

        if not os.path.exists(test_path_labels):
            os.makedirs(test_path_labels)
        if not os.path.exists(test_path_src):
            os.makedirs(test_path_src)
        if not os.path.exists(train_path_labels):
            os.makedirs(train_path_labels)
        if not os.path.exists(train_path_src):
            os.makedirs(train_path_src)

        boudaryTests = 0 if self.test_percentage == 0 else int(len(labels) / 100.0 * self.test_percentage)
        print "%d images will be splitted and used for tests" % boudaryTests

        trainSrc = shuffledSrc[boudaryTests:]
        trainLabels = shuffledLabels[boudaryTests:]

        testSrc = shuffledSrc[:boudaryTests:]
        testLabels = shuffledLabels[:boudaryTests]

        magenta = (255, 0, 255)
        idx_magenta = cv2.bitwise_not(self.mask) > 0

        print "Creating the training dataset"
        for i in range(len(trainSrc)):
            srcImg = FileManager.LoadImage(trainSrc[i], self.src_path)
            #srcImg = self.fillVoidClassInSource(srcImg, idx_magenta, magenta)
            lblImg = FileManager.LoadImage(trainLabels[i], self.labels_path, cv2.IMREAD_GRAYSCALE)

            resizedSrcImg = cv2.resize(srcImg, self.targetSize, interpolation=cv2.INTER_CUBIC)
            resizedLblImg = cv2.resize(lblImg, self.targetSize, interpolation=cv2.INTER_NEAREST)
            FileManager.SaveImage(resizedSrcImg, trainSrc[i], train_path_src)
            FileManager.SaveImage(resizedLblImg, trainLabels[i], train_path_labels)

        print "Creating the testing dataset"
        for i in range(len(testSrc)):
            srcImg = FileManager.LoadImage(testSrc[i], self.src_path)
            #srcImg = self.fillVoidClassInSource(srcImg, idx_magenta, magenta)
            lblImg = FileManager.LoadImage(testLabels[i], self.labels_path, cv2.IMREAD_GRAYSCALE)

            resizedSrcImg = cv2.resize(srcImg, self.targetSize, interpolation=cv2.INTER_CUBIC)
            resizedLblImg = cv2.resize(lblImg, self.targetSize, interpolation=cv2.INTER_NEAREST)
            FileManager.SaveImage(resizedSrcImg, testSrc[i], test_path_src)
            FileManager.SaveImage(resizedLblImg, testLabels[i], test_path_labels)


    def resizeImages(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_names = sorted([f for f in os.listdir(input_dir)])

        for i in range(len(img_names)):
            img = FileManager.LoadImage(img_names[i], input_dir)
            resizedSrcImg = cv2.resize(img, self.targetSize, interpolation=cv2.INTER_CUBIC)
            FileManager.SaveImage(resizedSrcImg, img_names[i], output_dir)
