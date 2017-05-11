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

if __name__ == '__main__':
    src_str = "0051.jpg"
    #mask = MaskCreator.create_circle_mask(1440)
    #src = FileManager.LoadImage(src_str, "images/src")

    #markers_fg, markers_bg = MarkersSelector.select_markers(src, mask, skeletonize=True)
    #FileManager.SaveImage(markers_bg, "background.png")
    #FileManager.SaveImage(markers_fg, "foreground.png")

    sky_view_factor_test()


