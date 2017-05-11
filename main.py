from core.OpticalRectifier import OpticalRectifier
from core.SkySegmentor import SkySegmentor
from core.DatasetManager import DatasetManager
from tools.FileManager import FileManager
from tools.MaskMerger import MasksMerger
from tools.ClassificationSelector import beginSelection
from tools.ImageDataGenerator import ImageDataGenerator

import numpy as np
import cv2
import os
import re
from shutil import copy
from cnn.cnn_main import main as cnn_main

width = 480
heigth = 480

def rectifyAllInputs(inputFolder, outputFolder):
    tableSrc = [24, 25, 25, 26, 26, 27, 27, 29, 29, 32, 32, 36, 36, 40, 41, 43, 44]
    imgViewAngle = 180
    imgWidth = 1440
    imgHeight = 1440

    oprec = OpticalRectifier(tableSrc, imgViewAngle, imgWidth, imgHeight)
    oprec.rectifyAllInputs(inputFolder, outputFolder)


def mergeMasks():
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    mm = MasksMerger("images/build/", "images/sky/", mask)
    mm.MergeAll()


def prepareDataset(dataset_output_path="./cnn/dataset", valid_percent=15, test_percent=15, resize_tests_images=False):
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    dmgr = DatasetManager(mask, valid_percent, test_percent, (width, heigth), dataset_output_path=dataset_output_path)
    #dmgr.split_dataset_by_mostly_represented_class("images/src", "images/annoted", mask)
    dmgr.create_synthetic_balanced_dataset_with_data_augmentation("images/src", "images/annoted", mask, 1440, 1440, 4)
    return []
    #if dmgr.checkForLabelsSanity() == 0:
    dmgr.create_annotated_images()
    dmgr.create_final_dataset()
    if resize_tests_images:
        dmgr.resize_images("/home/brandtk/Desktop/svf_samples", "./cnn/test_images/")
    return dmgr.classes_weigth

def prepareNewLabels(final_size, labels_path="images/newlabels", src_path="images/src", output_path="outputs/"):
    reg = r'\w+\.(jpg|jpeg|png)'
    labels = [f for f in os.listdir(labels_path) if re.match(reg, f.lower())]

    if not os.path.exists(os.path.join(output_path, "newlabels_src")):
        os.makedirs(os.path.join(output_path, "newlabels_src"))

    for lab in labels:
        img = FileManager.LoadImage(lab, labels_path)
        resized = cv2.resize(img, final_size, interpolation=cv2.INTER_NEAREST)
        FileManager.SaveImage(resized, lab, os.path.join(output_path, "newlabels"))
        src_name = ".".join([lab.split(".")[0], "jpg"])
        copy(os.path.join(src_path, src_name), os.path.join(output_path, "newlabels_src", src_name))

if __name__ == '__main__':
    class_weights = {0: 1.9991311197110881, 1: 4.768665483757782, 2: 9.548975463506991, 3: 5.39499062619272}
    class_weights = prepareDataset(resize_tests_images=False)
    #cnn_main(width, heigth, class_weights)

    #beginSelection("/home/brandtk/SVF-tocorrect/src", "/home/brandtk/SVF-tocorrect/pred", "outputs/")
    #prepareNewLabels((1440, 1440), "images/labels480x480", "/home/brandtk/Desktop/SVF/outputs_NE")
    #mergeMasks()