from core.OpticalRectifier import OpticalRectifier
from core.SkySegmentor import SkySegmentor
from tools.FileManager import FileManager
from core.Nagao import NagaoFilter
import os, ntpath
import re
import time

def rectifyOpticTest():
    tableSrc = [19, 20, 21, 23, 27, 27, 28, 28, 28]
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

    files = [f for f in os.listdir(inputFolder) if re.match(r'[0-9]+.*\.jpg', f)]
    alreadyTreated = [f for f in os.listdir(outputFolder) if re.match(r'[0-9]+.*\.jpg', f)]
    toTreat = list(set(files) - set(alreadyTreated))

    tableSrc = [19, 20, 21, 23, 27, 27, 28, 28, 28]
    imgViewAngle = 180
    scaleFactor = 1.2
    scaleOffset = 0

    oprec = OpticalRectifier(tableSrc, imgViewAngle, scaleFactor, scaleOffset)

    length = len(toTreat)
    i = 1
    ell_time = 0

    print "%d images remaining" % length
    for file in toTreat:
        startTime = time.time()
        print "%d/%d: processing file %s" % (i, length, file)
        imgSrc = FileManager.LoadImage(file)
        imgres = oprec.rectifyImage(imgSrc)
        print "%s/%s" %(outputFolder, path_leaf(file))
        FileManager.SaveImage(imgres, path_leaf(file))

        ell_time += time.time() - startTime
        #print "%f, %f" % ((ell_time / i), (length - i))
        print "remaining time estimate %d minutes" % (((ell_time / i) * (length - i)) / 60)

        i += 1

if __name__ == '__main__':
    #rectifyOpticTest()
    #segmentationKMeans()
    #segmentationByColor()
    #test()
    rectifyAllInputs("images/", "outputs")
