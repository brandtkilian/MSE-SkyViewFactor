from core.OpticalRectifier import OpticalRectifier
from core.SkySegmentor import SkySegmentor
from tools.FileManager import FileManager
from tools.MaskMerger import MasksMerger
import os, ntpath
import re
import time
import numpy as np
import cv2

def rectifyOpticTest():
    #tableSrc = [19, 20, 21, 23, 27, 27, 28, 28, 28]
    tableSrc = [0, 49, 49, 51, 56, 61, 69, 76, 84, 88, 93]
    imgViewAngle = 180
    scaleFactor = 1.2
    scaleOffset = 0

    oprec = OpticalRectifier(tableSrc, imgViewAngle, scaleFactor, scaleOffset)
    imgSrc = FileManager.LoadImage("03362o.jpg")

    imgres = oprec.rectifyImage(imgSrc)
    FileManager.SaveImage(imgres, "0001.jpg")

def segmentationByColor():
    ss = SkySegmentor()
    imgSrc = FileManager.LoadImage("0001.jpg")
    mask = ss.getSkyMaskByBlueColor(imgSrc, 210)

    FileManager.SaveImage(mask, "firstSkySegmentationTry_mask.jpg")

def segmentationKMeans():
    ss = SkySegmentor()
    imgSrc = FileManager.LoadImage("0001.jpg")

    for k in range(3, 4):
        res, center = ss.kMeansSegmentation(imgSrc, k)
        FileManager.SaveImage(res, "kmeansSegBgrK%d.jpg"%k)

def test():
    ss = SkySegmentor()
    imgSrc = FileManager.LoadImage("0001.jpg")
    res = ss.felzenszwalb(imgSrc)
    FileManager.SaveImage(res, "impressme.jpg")

def rectifyAllInputs(inputFolder, outputFolder):
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    files = [f for f in os.listdir(inputFolder) if re.match(r'[0-9]+.*\.jpg', f.lower())]
    alreadyTreated = [f for f in os.listdir(outputFolder) if re.match(r'[0-9]+.*\.jpg', f.lower())]
    toTreat = list(set(files) - set(alreadyTreated))

    tableSrc = [24, 25, 25, 26, 26, 27, 27, 29, 29, 32, 32, 36, 36, 40, 41, 43, 44]
    imgViewAngle = 180
    imgWidth = 1440
    imgHeight = 1440

    oprec = OpticalRectifier(tableSrc, imgViewAngle, imgWidth, imgHeight)

    length = len(toTreat)
    i = 1
    ell_time = 0

    print "%d images remaining" % length
    for file in sorted(toTreat):
        startTime = time.time()
        imgSrc = FileManager.LoadImage(file)
        imgres = oprec.rectifyImage(imgSrc)

        FileManager.SaveImage(imgres, path_leaf(file))
        ell_time += time.time() - startTime

        if i % 50 == 0:
            print "%s/%s" % (outputFolder, path_leaf(file))
            print "%d/%d: processed file %s" % (i, length, file)
            print "remaining time estimate %d minutes" % (((ell_time / i) * (length - i)) / 60)

        i += 1

def mergeMasks():
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    mm = MasksMerger("images/build/", "images/sky/", mask)
    mm.MergeAll()

if __name__ == '__main__':
    #rectifyOpticTest()
    #segmentationKMeans()
    #segmentationByColor()
    #test()
    #rectifyAllInputs("images/", "outputs")
    mergeMasks()
