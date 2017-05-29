import numpy as np
import math
from tools.FuncTimeProfiler import profile
import cv2
from tools.FileManager import FileManager
import os, ntpath
import re
import time


class OpticalRectifier:

    def __init__(self, tableSrc, imgViewAngle, imgWidth, imgHeight):
        self.tableSrc = tableSrc
        self.imgViewAngle = imgViewAngle
        self.mapping = None
        self.initialize_coords_transform(imgWidth, imgHeight)
        self.mask = np.zeros((imgHeight, imgWidth, 1), np.uint8)
        cv2.circle(self.mask, (imgWidth/2, imgHeight/2), imgWidth/2, (255, 255, 255), -1)


    @profile
    def rectify_image(self, img):
        rectified = img[self.mapping[..., 1], self.mapping[..., 0]]
        masked = cv2.bitwise_and(rectified, rectified, mask=self.mask)
        return masked

    @profile
    def initialize_coords_transform(self, width, height):
        if self.mapping is None:
            self.mapping = np.zeros((height, width, 2), np.int32)

            rI = width / 2.0
            xCenter = width / 2
            yCenter = height / 2

            tableCorr = self.get_rectified_calib_table(rI)
            summedTableSrc = [0] + [sum(self.tableSrc[:i+1]) for i in range(len(self.tableSrc))]

            f = np.poly1d(np.polyfit(tableCorr, np.asarray(summedTableSrc), 3))

            for yCor in range(height):
                for xCor in range(width):
                    dx = xCor - xCenter
                    dy = yCor - yCenter
                    rCor = math.sqrt(dx * dx + dy * dy)
                    angle = dx / rCor if rCor != 0 else 0
                    alpha = math.asin(angle)
                    if yCor > yCenter:
                        alpha = math.pi - alpha
                    rSrc = f(rCor)

                    if math.fabs(rSrc) < rI:
                        xSrc = int(xCenter + rSrc * math.sin(alpha))
                        ySrc = int(yCenter - rSrc * math.cos(alpha))
                        self.mapping[yCor, xCor, 0] = xSrc
                        self.mapping[yCor, xCor, 1] = ySrc

    def get_rectified_calib_table(self, rI):
        length = len(self.tableSrc)
        assert length > 0
        return np.asarray([0] + [(rI / length) * (i+1) for i in range(length)])

    def rectify_all_inputs(self, inputFolder, outputFolder):
        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        reg = r'\w+\.(jpg|gif|png)'
        files = [f for f in os.listdir(inputFolder) if re.match(reg, f.lower())]
        alreadyTreated = [f for f in os.listdir(outputFolder) if re.match(reg, f.lower())]
        toTreat = list(set(files) - set(alreadyTreated))

        length = len(toTreat)
        i = 1
        ell_time = 0

        print "%d images remaining" % length
        for file in sorted(toTreat):
            startTime = time.time()
            imgSrc = FileManager.LoadImage(file)
            imgres = self.rectify_image(imgSrc)

            FileManager.SaveImage(imgres, path_leaf(file), outputFolder)
            ell_time += time.time() - startTime

            if i % 50 == 0:
                print "%s/%s" % (outputFolder, path_leaf(file))
                print "%d/%d: processed file %s" % (i, length, file)
                print "remaining time estimate %d minutes" % (((ell_time / i) * (length - i)) / 60)

            i += 1
