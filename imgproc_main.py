from core.MarkersSelector import MarkersSelector
from tools.FileManager import FileManager
from tools.MaskCreator import MaskCreator
from core.SkySegmentor import SkySegmentor
from core.SkyViewFactorCalculator import SkyViewFactorCalculator
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from core.DatasetManager import DatasetManager
from core.BalancedImageDataGenerator import BalancedImageDataGenerator
from core.ImageDataGenerator import ImageDataGenerator
from itertools import izip
from core.ClassesEnum import Classes
import operator
import random

def segmentation_by_color():
    ss = SkySegmentor()
    imgSrc = FileManager.LoadImage("0051.jpg")
    mask = ss.get_sky_mask_by_blue_color(imgSrc, 210)

    FileManager.SaveImage(mask, "firstSkySegmentationTry_mask.jpg")

def segmentation_KMeans():
    ss = SkySegmentor()
    imgSrc = FileManager.LoadImage("0001.jpg")

    for k in range(3, 4):
        res, center = ss.kMeans_segmentation(imgSrc, k)
        FileManager.SaveImage(res, "kmeansSegBgrK%d.jpg"%k)

def sky_view_factor_test():
    src = FileManager.LoadImage("0001.jpg", "outputs/watershed", cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)

    path = "images/labels"
    labels = sorted([f for f in os.listdir(path)])[:]

    sky = []
    veg = []
    built = []
    for f in labels:
        img = FileManager.LoadImage(f, path)
        factors = SkyViewFactorCalculator.compute_factor_bgr_labels(img)
        sky.append(factors[0])
        veg.append(factors[1])
        built.append(factors[2])

    plt.subplot(111)
    plt.plot(sky, 'b', label="Sky")
    plt.plot(veg, 'g', label="Veg")
    plt.plot(built, 'r', label="Buildings")
    plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
               ncol=2, shadow=True, title="Legend", fancybox=True)
    plt.savefig("outputs/factors.jpg")


def test_balanced_generator():
    global length
    mask = MaskCreator.create_circle_mask(1440)
    dmgr = DatasetManager(mask, 0, 0, 1440)
    # dmgr.create_annotated_images()
    bidg = BalancedImageDataGenerator("images/src", "outputs/annoted", 480, 480, 4, seed=random.randint(0, 1999999),
                                      rotate=True)
    idg = ImageDataGenerator("images/src", "outputs/annoted", 1440, 1440, 4, seed=random.randint(0, 1999999))
    l = bidg.label_generator(binarized=False)
    i = 0
    averages = [0, 0, 0]
    tot = cv2.countNonZero(mask)
    for lbl_src in l:
        if i > 10000:
            break
        i += 1
        idx_sky = lbl_src == Classes.SKY
        idx_veg = lbl_src == Classes.VEGETATION
        idx_build = lbl_src == Classes.BUILT

        nb = idx_sky.sum() / float(tot)
        ng = idx_veg.sum() / float(tot)
        nr = idx_build.sum() / float(tot)

        percents = (nb, ng, nr)
        averages = map(operator.add, averages, percents)

        if i % 50 == 0:
            print "Currently generated %d" % i
    length = i
    averages = map(lambda x: x / length, averages)
    print bidg.occurences_dict
    print len(bidg.occurences_dict)
    print averages


if __name__ == '__main__':
    test_balanced_generator()



