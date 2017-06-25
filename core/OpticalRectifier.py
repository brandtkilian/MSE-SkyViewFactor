import numpy as np
import math
from tools.FuncTimeProfiler import profile
import cv2
from tools.FileManager import FileManager
import os, ntpath
import re
import time
from tools.MaskCreator import MaskCreator


class OpticalRectifier:
    """Correct the images optical projection according to calibration distances"""

    def __init__(self, tableSrc, imgViewAngle, imgWidth, imgHeight):
        self.tableSrc = tableSrc
        self.imgViewAngle = imgViewAngle
        self.mapping_x = None
        self.mapping_y = None
        self.initialize_coords_transform(imgWidth, imgHeight)
        self.mask = MaskCreator.create_circle_mask(imgWidth)


    @profile
    def rectify_image(self, img):
        rectified = cv2.remap(img, self.mapping_x, self.mapping_y, cv2.INTER_CUBIC)
        masked = cv2.bitwise_and(rectified, rectified, mask=self.mask)

        return masked

    @profile
    def initialize_coords_transform(self, width, height):
        if self.mapping_x is None or self.mapping_y is None:
            self.mapping_x = np.zeros((height, width), np.float32)
            self.mapping_y = np.zeros((height, width), np.float32)

            rI = width / 2.0
            xCenter = width / 2
            yCenter = height / 2

            tableCorr = self.get_rectified_calib_table(rI)
            summedTableSrc = [0] + [sum(self.tableSrc[:i+1]) for i in range(len(self.tableSrc))]

            f = np.poly1d(np.polyfit(tableCorr, np.asarray(summedTableSrc), 2))

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
                        xSrc = xCenter + rSrc * math.sin(alpha)
                        ySrc = yCenter - rSrc * math.cos(alpha)
                        self.mapping_x[yCor, xCor] = xSrc
                        self.mapping_y[yCor, xCor] = ySrc

    def get_rectified_calib_table(self, rI):
        length = len(self.tableSrc)
        assert length > 0
        return np.asarray([0] + [(rI / length) * (i+1) for i in range(length)])

    def rectify_all_inputs(self, input_folder, output_folder):
        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        reg = r'\w+\.(jpg|gif|png)'
        files = [f for f in os.listdir(input_folder) if re.match(reg, f.lower())]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        already_processed = [f for f in os.listdir(output_folder) if re.match(reg, f.lower())]
        to_process = list(set(files) - set(already_processed))

        length = len(to_process)
        i = 1
        ell_time = 0

        print "%d images remaining" % length
        for file in sorted(to_process):
            startTime = time.time()
            img_src = FileManager.LoadImage(file, input_folder)
            img_res = self.rectify_image(img_src)

            FileManager.SaveImage(img_res, path_leaf(file), output_folder)
            ell_time += time.time() - startTime

            if i % 50 == 0:
                print "%s/%s" % (output_folder, path_leaf(file))
                print "%d/%d: processed file %s" % (i, length, file)
                print "remaining time estimate %d minutes" % (((ell_time / i) * (length - i)) / 60)

            i += 1
