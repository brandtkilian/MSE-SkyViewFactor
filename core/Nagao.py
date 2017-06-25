import cv2
import numpy as np


class NagaoFilter:
    """Class that encapsulate the creation of
     Nagao kernels and methods to apply the filter on a image"""

    def __init__(self, kernelSize):
        self.setKernelSize(kernelSize)

    def setKernelSize(self, kernelSize):
        kernelSize = -kernelSize if kernelSize < 0 else kernelSize
        self.kernelSize = kernelSize
        kernelsF, kernelsI = self.initKernels(kernelSize)
        self.kernelsF = kernelsF
        self.kernelsI = kernelsI

    def initKernels(self, kernelSize):

        if kernelSize % 2 == 0:
            print "kernel_size must be odd: ", (kernelSize + 1)
            kernelSize += 1

        halfSize = (kernelSize - 1) / 2
        idx = np.arange(kernelSize ** 2)
        rowIndexes = idx / kernelSize
        colIndexes = idx % kernelSize

        rowIndexes_centered = rowIndexes - halfSize
        colIndexes_centered = colIndexes - halfSize

        kernelsI = []
        kernelsF = []
        for kernelIdx in range(0, 9):
            kernel = np.zeros(kernelSize ** 2, np.uint8)

            if kernelIdx == 0:
                condR = rowIndexes <= halfSize
                condC = colIndexes <= halfSize

            elif kernelIdx == 1:
                condR = rowIndexes <= halfSize
                condC = colIndexes >= halfSize

            elif kernelIdx == 2:
                condR = rowIndexes >= halfSize
                condC = colIndexes >= halfSize

            elif kernelIdx == 3:
                condR = rowIndexes >= halfSize
                condC = colIndexes <= halfSize

            elif kernelIdx == 4:
                condR = (rowIndexes >= np.floor(kernelSize / 4.0)) & (rowIndexes < 3.0 * kernelSize / 4.0)
                condC = (colIndexes >= np.floor(kernelSize / 4.0)) & (colIndexes < 3.0 * kernelSize / 4.0)

            elif kernelIdx == 5:
                condR = rowIndexes_centered <= 0
                condC = np.abs(colIndexes_centered) <= -rowIndexes_centered

            elif kernelIdx == 6:
                condC = colIndexes_centered >= 0
                condR = np.abs(rowIndexes_centered) <= colIndexes_centered

            elif kernelIdx == 7:
                condR = rowIndexes_centered >= 0
                condC = np.abs(colIndexes_centered) <= rowIndexes_centered

            elif kernelIdx == 8:
                condC = colIndexes_centered <= 0
                condR = np.abs(rowIndexes_centered) <= -colIndexes_centered

            # set mask to 1
            selected_idx = idx[condR & condC]
            kernel[selected_idx] = 1
            kernel = np.reshape(kernel, (kernelSize, kernelSize))
            ksum = 1.0 * kernel.sum()
            kernelNormalized = kernel / ksum
            kernelsF.append(kernelNormalized)
            kernelsI.append(kernel)

        return kernelsF, kernelsI

    def executeKernels(self, img):
        imgsSmoothed = []

        for kernel in self.kernelsF:
            imgConv = cv2.filter2D(img, -1, kernel)
            imgsSmoothed.append(imgConv)

        return imgsSmoothed


    def computeMinMax(self, img):
        imgsMin = []
        imgsMax = []

        for i in range(0, len(self.kernelsF)):
            img_min = cv2.morphologyEx(img, cv2.MORPH_ERODE, self.kernelsI[i])  # None, None, iterations)
            imgsMin.append(img_min)
            img_max = cv2.morphologyEx(img, cv2.MORPH_DILATE, self.kernelsI[i])  # None, None, iterations)
            imgsMax.append(img_max)

        return imgsMin, imgsMax

    def merge(self, imgsSmoothed, imgsMin, imgsMax):

        imageA = imgsSmoothed[0]
        minimA = imgsMin[0]
        maximA = imgsMax[0]
        gradiA = maximA - minimA

        for i in range(1, len(imgsSmoothed)):
            imageB = imgsSmoothed[i]
            minimB = imgsMin[i]
            maximB = imgsMax[i]

            gradiB = maximB - minimB
            idxes = gradiB < gradiA

            imageA[idxes] = imageB[idxes]
            minimA[idxes] = minimB[idxes]
            maximA[idxes] = maximB[idxes]
            gradiA[idxes] = gradiB[idxes]

        return imageA # filtered

    def filter(self, img):
        imgsSmoothed = self.executeKernels(img)
        imgsMin, imgsMax = self.computeMinMax(img)
        filtered = self.merge(imgsSmoothed, imgsMin, imgsMax)

        return filtered