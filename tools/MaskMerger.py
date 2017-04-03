import cv2
import glob
import ntpath
from tools.FileManager import FileManager
import matplotlib.pyplot as plt
import numpy as np


class MasksMerger():

    def __init__(self, buildPath, skyPath, mask):
        self.path1 = buildPath
        self.path2 = skyPath
        self.mask = mask


    def MergeAll(self, outputDir="outputs/"):
        files1 = [f for f in glob.iglob("%s/*.png" % self.path1)]
        files2 = [f for f in glob.iglob("%s/*.png" % self.path2)]

        print len(files1), len(files2)
        assert len(files1) == len(files2)

        for i in range(len(files1)):
            red = cv2.imread(files1[i], flags=cv2.IMREAD_GRAYSCALE)
            blue = cv2.imread(files2[i], flags=cv2.IMREAD_GRAYSCALE)

            green = cv2.bitwise_not(cv2.bitwise_or(red, blue))
            merged = cv2.merge((blue, green, red))
            height, width, channels = merged.shape

            (cnts, _) = cv2.findContours(green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c, False) <= 100:
                    rect = cv2.boundingRect(c)
                    sY = rect[1] - rect[3] - 1
                    sY = sY if sY >= 0 else 0
                    eY = rect[1] + 3 * rect[3] + 1
                    eY = eY if eY <= height else height
                    sX = rect[0] - rect[2] - 1
                    sX = sX if sX > 0 else 0
                    eX = rect[0] + 3 * rect[2] + 1
                    eX = eX if eX < width else width
                    crop = merged[sY:eY, sX:eX]
                    bc, rc, gc = cv2.split(crop)
                    nb = cv2.countNonZero(bc)
                    ng = cv2.countNonZero(gc)
                    nr = cv2.countNonZero(rc)
                    counts = (nb, nr)
                    colors = [(255, 0, 0), (0, 0, 255)]
                    winner = sorted(enumerate(counts), reverse=True)
                    idx, count = winner[0]
                    color = colors[idx]
                    cv2.drawContours(merged, [c], 0, color, -1)

            b, g, r = cv2.split(merged)
            conflicts = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_and(r, b),
                                       cv2.bitwise_and(r, g)), cv2.bitwise_and(b, r))
            idx = conflicts > 0
            merged[idx] = (255, 255, 255)
            merged = cv2.bitwise_and(merged, merged, None, self.mask)
            FileManager.SaveImage(merged, ntpath.basename(files1[i]))