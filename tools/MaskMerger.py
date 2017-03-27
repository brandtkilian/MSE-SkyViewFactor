import cv2
import glob
import ntpath
from tools.FileManager import FileManager
from core.Nagao import NagaoFilter


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
            maskedMerged = cv2.bitwise_and(merged, merged, mask=self.mask)

            FileManager.SaveImage(maskedMerged, ntpath.basename(files1[i]))