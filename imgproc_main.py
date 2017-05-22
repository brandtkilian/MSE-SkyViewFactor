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
from core.ImageDataGenerator import ImageDataGenerator, PossibleTransform, NormType
from itertools import izip
from core.ClassesEnum import Classes
import operator
import random
from tools.BSPMarkerCreator import BSPMarkerCreator


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
    img_names = ["0001.jpg", "0031.jpg", "0051.jpg"]
    for img_name in img_names:
        src = FileManager.LoadImage(img_name, "outputs/watershed", cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)

        factor = SkyViewFactorCalculator.compute_factor(thresh)
        print "SVF for image %s = %.3f" % (img_name, factor)

    print "black image test"
    black = np.zeros((1440, 1440), np.uint8)
    factor = SkyViewFactorCalculator.compute_factor(black)
    print "svf for black image : %.3f" % factor
    print "white image test"
    white = np.ones((1440, 1440), np.uint8)
    white *= 255
    factor = SkyViewFactorCalculator.compute_factor(white)
    print "svf for white image : %.3f" % factor


def sky_view_factor_angle_test():
    img_names = ["0001.jpg", "0031.jpg", "3481.jpg"]
    for img_name in img_names:
        src = FileManager.LoadImage(img_name, "outputs/watershed", cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)

        factor = SkyViewFactorCalculator.compute_sky_angle_estimation(thresh, (720, 720), 0, 720)
        cv2.circle(src, (720, 720), int(factor * 720 / 90.0), (255, 0, 0), 3)
        FileManager.SaveImage(src, "circle" + img_name)
        print "Angle from vertical including 50%% of svf for image %s = %.3f" % (img_name, factor)


def test_svf_algorithm():
    quarter = np.zeros((1440, 1440), np.uint8)
    cv2.rectangle(quarter, (0, 0), (720, 720), (255, 255, 255), -1)
    factor = SkyViewFactorCalculator.compute_factor(quarter, number_of_steps=120)
    diff = abs(factor - 0.25)
    assert diff < 1e-3, "Quarter SVF should be equal to 0.25, algorithm returned %.5f, (diff: %.5f)" % (factor, diff)

    half = np.zeros((1440, 1440), np.uint8)
    cv2.rectangle(half, (0, 0), (1440, 720), (255, 255, 255), -1)
    factor = SkyViewFactorCalculator.compute_factor(half)
    diff = abs(factor - 0.50)
    assert diff < 1e-3, "Half SVF should be equal to 0.50, algorithm returned %.5f, (diff: %.5f)" % (factor, diff)

    print "tests succeed !"


def svf_graphs():
    path = "/home/brandtk/predictions2017-05-17_12:57:48"
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

    transforms = [(PossibleTransform.GaussianNoise, 0.15),
                  (PossibleTransform.Sharpen, 0.15),
                  (PossibleTransform.MultiplyPerChannels, 0.15),
                  (PossibleTransform.AddSub, 0.15),
                  (PossibleTransform.Multiply, 0.15), ]

    bidg = BalancedImageDataGenerator("images/src", "outputs/annoted", 480, 480, 4, allow_transforms=True,
                                               rotate=True, transforms=transforms,
                                               lower_rotation_bound=0, higher_rotation_bound=360, magentize=True, seed=random.randint(1, 10e6), yield_names=True)

    idg = ImageDataGenerator("images/src", "outputs/annoted", 1440, 1440, 4, seed=random.randint(0, 1999999))
    l = bidg.label_generator(binarized=False)

    gen = izip(bidg.image_generator(roll_axis=False), bidg.label_generator(binarized=False))
    i = 0
    averages = [0, 0, 0]
    tot = cv2.countNonZero(mask)
    for img_info, lbl_info in gen:
        if i > 10000:
            break
        i += 1
        img = img_info[0]
        img_name = img_info[1]
        lbl_src = lbl_info[0]
        lbl_name = lbl_info[1]
        idx_sky = lbl_src == Classes.SKY
        idx_veg = lbl_src == Classes.VEGETATION
        idx_build = lbl_src == Classes.BUILT

        mask_sky = np.zeros(lbl_src.shape, np.uint8)
        mask_sky[idx_sky] = 255
        cv2.imshow("x", cv2.bitwise_and(img, img, mask=mask_sky))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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


def svf_graph_and_mse():
    path_gt = "/home/brandtk/MSE-SkyViewFactor/images/svf/gt"
    path_pred = "/home/brandtk/MSE-SkyViewFactor/images/svf/preds"

    gt = sorted([f for f in os.listdir(path_gt)])[:]
    preds = sorted([f for f in os.listdir(path_pred)])[:]

    true_sky = []
    true_veg = []
    true_built = []

    for f in gt:
        img = FileManager.LoadImage(f, path_gt, flags=cv2.IMREAD_GRAYSCALE)
        factors = SkyViewFactorCalculator.compute_factor_annotated_labels(img, center=(240, 240), radius=240)
        true_sky.append(factors[0])
        true_veg.append(factors[1])
        true_built.append(factors[2])

    pred_sky = []
    pred_veg = []
    pred_built = []

    for f in preds:
        img = FileManager.LoadImage(f, path_pred)
        factors = SkyViewFactorCalculator.compute_factor_bgr_labels(img, center=(240, 240), radius=240)
        pred_sky.append(factors[0])
        pred_veg.append(factors[1])
        pred_built.append(factors[2])

    to_compute = [(true_sky, pred_sky), (true_veg, pred_veg), (true_built, pred_built)]

    mses = []
    for true, pred in to_compute:
        mse = SkyViewFactorCalculator.compute_mean_square_error(true, pred)
        mses.append(mse)

    print "MeanSquaredError: Sky: %.3E, Veg: %.3E, Built: %.3E" % tuple(mses)

    exp_x = ()

    labels = ["Sky", "Vegetation", "Building"]
    styles = ['bo', 'g^', 'rs']
    i = 0
    for true, pred in to_compute:
        plt.subplot(111)
        plt.plot(true, pred, styles[i])
        plt.suptitle("%s view factor (MSE=%.3E)" % (labels[i], mses[i]))
        plt.xlabel("True")
        plt.ylabel("Predict")
        plt.axis((0, 1, 0, 1))
        plt.plot([0, 1], [0, 1], ls="--", c=".2")
        plt.savefig("outputs/mse_%s.jpg" % labels[i])
        plt.cla()
        i += 1

if __name__ == '__main__':
    svf_graphs()
    #test_balanced_generator()
    #sky_view_factor_test()
    #sky_view_factor_angle_test()
    #test_svf_algorithm()
    #BSPMarkerCreator.create_markers(foreground_channel=2)



