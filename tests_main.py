import os
import re
from shutil import copy

import cv2
import numpy as np

from cnn.cnn_main import main as cnn_main
from core.DatasetManager import DatasetManager
from core.OpticalRectifier import OpticalRectifier
from tools.FileManager import FileManager
from tools.MaskMerger import MasksMerger


def rectify_all_inputs(inputFolder, outputFolder):
    """Using the optical rectifier, rectify the optic of an entire directory"""
    tableSrc = [24, 25, 25, 26, 26, 27, 27, 29, 29, 32, 32, 36, 36, 40, 41, 43, 44]
    imgViewAngle = 180
    imgWidth = 1440
    imgHeight = 1440

    oprec = OpticalRectifier(tableSrc, imgViewAngle, imgWidth, imgHeight)
    oprec.rectify_all_inputs(inputFolder, outputFolder)


def merge_masks():
    """Merge mask coming from IST following a strategy described in the report"""
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    MasksMerger.merge_from_sky_and_build("images/build/", "images/sky/", mask, "outputs/merged_masks")


def prepare_dataset(dataset_output_path="./cnn/dataset", valid_percent=1, test_percent=1, resize_tests_images=False):
    """Prepare the dataset if required following the splitting percentages."""
    dmgr = DatasetManager(valid_percent, test_percent, (1440, 1440), dataset_output_path=dataset_output_path)
    #dmgr.split_dataset_by_mostly_represented_class("images/src", "images/annoted", mask)
    #dmgr.create_synthetic_balanced_dataset_with_data_augmentation("images/src", "images/annoted", mask, 1440, 1440, 4)
    #if dmgr.checkForLabelsSanity() == 0:
    dmgr.create_annotated_images()  # the dataset manager is responsible of creating annotated images from rgb labels images
    dmgr.create_final_dataset()  # this method split the dataset into the right folder hierarchy
    if resize_tests_images:
        dmgr.resize_images("/home/brandtk/Desktop/svf_samples", "./cnn/test_images/")
    return dmgr.classes_weigth


def prepare_new_labels(final_size, labels_path="images/newlabels", src_path="images/src", output_path="outputs/"):
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

# main that prepapre the dataset if needed and launch the cnn_main method
if __name__ == '__main__':
    class_weights = {0: 3.219805224143668, 1: 5.691272269866311, 2: 3.332547924795312, 3: 4.68068666683757}
    new_class_weights = prepare_dataset(resize_tests_images=False)
    if sum(new_class_weights.itervalues()) > 0:
        class_weights = new_class_weights
        print "New class weights"
    class_weights = [v for v in class_weights.itervalues()]
    cnn_main(class_weights)

    #beginSelection("/home/brandtk/SVF-tocorrect/src", "/home/brandtk/SVF-tocorrect/pred", "outputs/")
    #prepare_new_labels((1440, 1440), "images/labels480x480", "/home/brandtk/Desktop/SVF/outputs_NE")
    #merge_masks()
