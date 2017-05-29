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
from tools.MaskMerger import MasksMerger
from tools.ClassificationSelector import beginSelection


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

    little_square = np.zeros((1440, 1440), np.uint8)
    cv2.rectangle(little_square, (300, 300), (400, 400), (255, 255, 255), -1)
    factor = SkyViewFactorCalculator.compute_factor(little_square, number_of_steps=120)

    little_square2 = np.zeros((1440, 1440), np.uint8)
    cv2.rectangle(little_square2, (670, 670), (770, 770), (255, 255, 255), -1)
    factor2 = SkyViewFactorCalculator.compute_factor(little_square2, number_of_steps=120)

    diff = abs(factor - factor2)
    assert diff > 1e-2, "SVF should not be equal between the two squares, (diff: %.5f, fact1=%.5f, fact2=%.5f)" % (diff, factor, factor2)

    totradius = 720
    radius1 = 700.0
    radius2 = 720.0
    r1 = radius1 / totradius
    r2 = radius2 / totradius
    angle1 = 0.25 * math.pi * (1 - r1)
    angle2 = 0.25 * math.pi * (1 - r2)
    print angle1, angle2, r1, r2
    h1 = (1 - r1) * math.tan(angle1)
    h2 = (1 - r2) * math.tan(angle2)

    print h1, h2
    h = abs(h2 - h1)
    r = (r1 + r2) / 2
    print h, r

    radius = (radius1 + radius2) / 2
    thickness = int(abs(radius2 - radius1))
    spheric_segment_area = 2 * math.pi * r * h
    spheric_segment_image = np.zeros((1440, 1440), np.uint8)

    cv2.circle(spheric_segment_image, (720, 720), int(radius), (255, 255, 255), thickness)
    factor = SkyViewFactorCalculator.compute_factor(spheric_segment_image)

    cv2.imshow("asad", spheric_segment_image)
    cv2.waitKey(0)

    assert abs(spheric_segment_area - factor) < 1e-3, "The area of the spheric segment computed with math formula approximation" \
                                                      " isn't equal to the View factor computed with the iterative algortihm (%.3f, %.3f)" %(spheric_segment_area, factor)

    print "tests succeed !"


def svf_graphs():
    path = "/home/brandtk/predictions2017-05-17_12:57:48"
    labels = sorted([f for f in os.listdir(path)])[:]
    sky = []
    veg = []
    built = []
    for f in labels[:]:
        img = FileManager.LoadImage(f, path)
        factors = SkyViewFactorCalculator.compute_factor_bgr_labels(img, center=(240, 240), radius=240)
        sky.append(factors[0])
        veg.append(factors[1])
        built.append(factors[2])
    plt.figure(figsize=(20, 6))
    plt.title("View Factors")
    plt.plot(sky, 'b', label="Sky")
    plt.plot(veg, 'g', label="Vegetation")
    plt.plot(built, 'r', label="Buildings")
    plt.ylim([0,1])
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
    path_gt = "cnn/dataset/tests/labels"
    path_pred = "cnn/evaluations/predictions2017-05-17_12:57:48/predictions"

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


def test_segmentation_watershed():
    path_gt = "images/gt_ws"
    path_src = "images/src_ws"

    gt = sorted([f for f in os.listdir(path_gt)])[:5]
    srcs = sorted([f for f in os.listdir(path_src)])[:5]

    sources = zip(gt, srcs)

    mask = MaskCreator.create_circle_mask(1440)

    svfs_pred = []
    svfs_pred_color = []
    svfs_gt = []

    segmentor = SkySegmentor()
    for gt, src in sources:
        img = FileManager.LoadImage(src, path_src)
        gt_img = FileManager.LoadImage(gt, path_gt, cv2.IMREAD_GRAYSCALE)

        sky_mask = segmentor.segment_watershed(img, mask, bluring_size=11, skeletonize=False, use_nagao=False)
        svf_pred = SkyViewFactorCalculator.compute_factor(sky_mask)
        sky_mask_color = segmentor.get_sky_mask_by_blue_color(img, 210)
        svf_pred_color = SkyViewFactorCalculator.compute_factor(sky_mask_color)

        sky_mask_gt = np.zeros(gt_img.shape, np.uint8)
        sky_mask_gt[gt_img == 1] = 255
        svf_gt = SkyViewFactorCalculator.compute_factor(sky_mask_gt)

        variance = segmentor.compute_variance(img,window_size=11)
        norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("norm", norm)
        cv2.waitKey(0)


        svfs_pred.append(svf_pred)
        svfs_gt.append(svf_gt)
        svfs_pred_color.append(svf_pred_color)
        FileManager.SaveImage(sky_mask, src, "outputs/watershedtests/pred")
        FileManager.SaveImage(sky_mask_gt, gt, "outputs/watershedtests/gt")
        FileManager.SaveImage(sky_mask_color, src, "outputs/watershedtests/color")

    mse = SkyViewFactorCalculator.compute_mean_square_error(svfs_gt, svfs_pred)
    plt.subplot(111)
    plt.plot(svfs_gt, svfs_pred, "bo")
    plt.suptitle("Sky view factor (MSE=%.3E)" % mse)
    plt.xlabel("True")
    plt.ylabel("Predict")
    plt.axis((0, 1, 0, 1))
    plt.plot([0, 1], [0, 1], ls="--", c=".2")
    plt.savefig("outputs/watershedtests/mse.jpg")
    plt.cla()
    plt.subplot(111)
    plt.plot(svfs_gt, svfs_pred_color, "bo")
    plt.suptitle("Sky view factor (MSE=%.3E)" % mse)
    plt.xlabel("True")
    plt.ylabel("Predict")
    plt.axis((0, 1, 0, 1))
    plt.plot([0, 1], [0, 1], ls="--", c=".2")
    plt.savefig("outputs/watershedtests/mse_color.jpg")



if __name__ == '__main__':
    #svf_graph_and_mse()
    svf_graphs()
    #test_balanced_generator()
    #sky_view_factor_test()
    sky_view_factor_angle_test()
    #test_svf_algorithm()

    folds = ["sky", "veg", "built"]

    mask = MaskCreator.create_circle_mask(1440)
    #for i, n in enumerate(folds):
    #    BSPMarkerCreator.create_markers(mask, "images/predictions", "outputs/markers_%s" % n, foreground_channel=i, skeletonize=False)
    #test_segmentation_watershed()

    #MasksMerger.merge_masks_from_all("images/tomerge/sky", "images/tomerge/veg", "images/tomerge/built", mask, "outputs/merged")
    #MasksMerger.merge_from_sky_and_build("images/build", "images/sky", mask, "outputs/megerino")
    beginSelection("images/SVF-dataset-150/src", "images/SVF-dataset-150/labels", "outputs/default")




