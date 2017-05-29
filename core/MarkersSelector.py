from core.Nagao import NagaoFilter
import cv2
import numpy as np
from tools.FileManager import FileManager


class MarkersSelector():

    @staticmethod
    def skeletonization(binary):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        done = False
        size = np.size(binary)
        skel = np.zeros(binary.shape, np.uint8)

        while not done:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary = eroded.copy()

            zeros = size - cv2.countNonZero(binary)
            if zeros == size:
                done = True

        return skel

    @staticmethod
    def select_markers(bgr, mask, skeletonize=False, eroding_size=31):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        sky_idx1 = np.logical_and(s < 13, v > 216)
        sky_idx2 = np.logical_and(np.logical_and(np.logical_and(s < 25, v > 204), h > 90), h < 150)
        sky_idx3 = np.logical_and(np.logical_and(np.logical_and(s < 128, v > 153), h > 100), h < 130)
        sky_idx4 = np.logical_and(np.logical_and(v > 88, h > 110), h < 120)

        idxes = [sky_idx1, sky_idx2, sky_idx3, sky_idx4]

        shape = bgr.shape
        potential_sky = np.zeros((shape[1], shape[0]), np.uint8)
        i = 0
        for idx in idxes:
            zeros = np.zeros(shape, np.uint8)
            zeros[idx] = bgr[idx]
            i += 1
            potential_sky[idx] = 255

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eroding_size, eroding_size))
        potential_sky_inv = cv2.bitwise_and(cv2.bitwise_not(potential_sky), mask)

        if skeletonize:
            return MarkersSelector.skeletonization(potential_sky), MarkersSelector.skeletonization(potential_sky_inv)
        else:
            return cv2.erode(potential_sky, ker), cv2.erode(potential_sky_inv, ker)

    @staticmethod
    def select_markers_otsu(bgr, mask, skeletonize=False, eroding_size=31, bluring_size=7, use_nagao=False):
        b, g, r = cv2.split(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        bluring_size = bluring_size if bluring_size % 2 != 0 else bluring_size + 1

        if use_nagao:
            nf = NagaoFilter(bluring_size)
            gray = nf.filter(gray)
        else:
            gray = cv2.blur(gray, (bluring_size, bluring_size))


        #FileManager.SaveImage(gray, "noiseless.png")
        idx = mask > 0
        tmpGray = gray[idx]

        threshold, thresh = cv2.threshold(tmpGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        #FileManager.SaveImage(thresh, "otsu.png")
        #FileManager.SaveImage(dummy, "dummy.png")

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eroding_size, eroding_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ker)
        thresh_inv = cv2.bitwise_and(cv2.bitwise_not(thresh), mask)
        #FileManager.SaveImage(thresh, "otsu_opened.png")

        if skeletonize:
            return MarkersSelector.skeletonization(thresh), MarkersSelector.skeletonization(thresh_inv)
        else:
            return cv2.erode(thresh, ker), cv2.erode(thresh_inv, ker)


