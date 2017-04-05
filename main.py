from core.OpticalRectifier import OpticalRectifier
from core.SkySegmentor import SkySegmentor
from core.DatasetManager import DatasetManager
from tools.FileManager import FileManager
from tools.MaskMerger import MasksMerger

import numpy as np
import cv2


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

def rectifyAllInputs(inputFolder, outputFolder):
    tableSrc = [24, 25, 25, 26, 26, 27, 27, 29, 29, 32, 32, 36, 36, 40, 41, 43, 44]
    imgViewAngle = 180
    imgWidth = 1440
    imgHeight = 1440

    oprec = OpticalRectifier(tableSrc, imgViewAngle, imgWidth, imgHeight)
    oprec.rectifyAllInputs(inputFolder, outputFolder)


def mergeMasks():
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    mm = MasksMerger("images/build/", "images/sky/", mask)
    mm.MergeAll()

def prepareDataset():
    mask = np.zeros((1440, 1440, 1), np.uint8)
    cv2.circle(mask, (1440 / 2, 1440 / 2), 1440 / 2, (255, 255, 255), -1)
    dmgr = DatasetManager(mask, 10, (360, 360))
    if dmgr.checkForLabelsSanity() == 0:
        dmgr.createAnotedImages()
        dmgr.createFinalDataset()

if __name__ == '__main__':
    #segmentationKMeans()
    #segmentationByColor()
    #test()
    #rectifyAllInputs("images/", "outputs")
    #mergeMasks()
    prepareDataset()
