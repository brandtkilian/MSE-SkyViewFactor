import cv2
import numpy as np
from core.Nagao import NagaoFilter
from tools.FileManager import FileManager
import skimage.segmentation, skimage.color


class SkySegmentor:

    def getSkyMaskByBlueColor(self, BGRImage, BlueT):
        eq = self.equalizeHist(BGRImage)
        FileManager.SaveImage(eq, "equalizedSrc.jpg")
        nagF = NagaoFilter(11)
        filtered = nagF.filter(eq)

        cv2.imwrite("outputs/nagao.jpg", filtered)

        amplifiedBlue = self.amplify(filtered, 0, 2, BlueT * 3 / 4)
        FileManager.SaveImage(amplifiedBlue, "ampB.jpg")
        b, g, r = cv2.split(amplifiedBlue)

        _, mask = cv2.threshold(b, BlueT, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask, iterations=1)

        FileManager.SaveImage(mask, "mask.jpg")

        return mask

    def kMeansSegmentation(self, img, K=4):
        img = self.equalizeHist(img)
        FileManager.SaveImage(img, "equalized.jpg")
        nf = NagaoFilter(51)
        img = nf.filter(img)
        img = self.amplifyHsv(img, 15, 30)
        FileManager.SaveImage(img, "nagao.jpg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        Z = hsv.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = cv2.cvtColor(res.reshape((img.shape)), cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        output = cv2.medianBlur(gray, 21)
        return output, center

    def felzenszwalb(self, img):
        #img = self.equalizeHist(img)
        #nf = NagaoFilter(11)
        #img = nf.filter(img)
        labels = skimage.segmentation.felzenszwalb(img, scale=1, sigma=6, min_size=300)
        return skimage.color.label2rgb(labels)


    def equalizeHist(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return img_output

    def equalizeHistClahe(self, img):
        clahe = cv2.createCLAHE()
        channels = cv2.split(img)
        claheified = []
        for c in channels:
            claheified.append(clahe.apply(c))

        return cv2.merge(claheified)

    def amplify(self, img, channel, factor, thresh):
        factor = factor if factor >= 1 else 1
        channels = cv2.split(img)
        c = channels[channel]
        cuint64 = c.astype(np.uint64)
        idx = cuint64 > thresh
        cuint64[idx] *= factor
        amplified = cuint64
        norm = cv2.normalize(amplified, None, np.min(c), 255, cv2.NORM_MINMAX)
        channels[channel] = norm.astype(np.uint8)
        img = cv2.merge(channels)
        return img

    def amplifyHsv(self, img, huemin, huemax):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        idx = h >= huemin
        idx2 = h > huemax
        idx -= idx2
        s[idx] = 255
        v[idx] = 255
        amplified = cv2.merge((h, s, v))
        return cv2.cvtColor(amplified, cv2.COLOR_HSV2BGR)
