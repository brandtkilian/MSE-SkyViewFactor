import cv2
import numpy as np
from core.Nagao import NagaoFilter
from tools.FileManager import FileManager
from tools.MaskCreator import MaskCreator
from core.MarkersSelector import MarkersSelector
import skimage.segmentation, skimage.color


class SkySegmentor():
    """Static class for segmenting sky from bgr images"""

    @staticmethod
    def get_sky_mask_by_blue_color(BGRImage, BlueT):
        eq = SkySegmentor.equalize_hist(BGRImage)
        FileManager.SaveImage(eq, "equalizedSrc.jpg")
        nagF = NagaoFilter(11)
        filtered = nagF.filter(eq)

        amplifiedBlue = SkySegmentor.amplify(filtered, 0, 2, BlueT * 3 / 4)
        b, g, r = cv2.split(amplifiedBlue)

        _, mask = cv2.threshold(b, BlueT, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask, iterations=1)

        return mask

    @staticmethod
    def kMeans_segmentation(self, img, K=4):
        img = SkySegmentor.equalize_hist(img)
        nf = NagaoFilter(51)
        img = nf.filter(img)
        img = self.amplifyHsv(img, 15, 30)
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

    @staticmethod
    def felzenszwalb(img):
        #img = self.equalizeHist(img)
        #nf = NagaoFilter(11)
        #img = nf.filter(img)
        labels = skimage.segmentation.felzenszwalb(img, scale=1, sigma=6, min_size=300)
        return skimage.color.label2rgb(labels)

    @staticmethod
    def equalize_hist(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return img_output

    @staticmethod
    def equalize_hist_Clahe(img):
        width = img.shape[1] / 30
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(width, width))
        channels = cv2.split(img)
        claheified = []
        for c in channels:
            claheified.append(clahe.apply(c))

        return cv2.merge(claheified)

    @staticmethod
    def amplify(img, channel, factor, thresh):
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

    @staticmethod
    def amplifyHsv(img, huemin, huemax):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        idx = h >= huemin
        idx2 = h > huemax
        idx -= idx2
        s[idx] = 255
        v[idx] = 255
        amplified = cv2.merge((h, s, v))
        return cv2.cvtColor(amplified, cv2.COLOR_HSV2BGR)

    @staticmethod
    def segment_watershed(bgr, mask, skeletonize=False, eroding_size=31, equalize=True, bluring_size=7, use_nagao=False):
        if equalize:
            bgr = SkySegmentor.equalize_hist(bgr)

        markersFG, markersBG = MarkersSelector.select_markers_otsu(bgr, mask, skeletonize=skeletonize,
                                                                   eroding_size=eroding_size, bluring_size=bluring_size,
                                                                   use_nagao=use_nagao)

        #markersFG, markersBG = MarkersSelector.select_markers(bgr, mask)

        markers = np.zeros(markersBG.shape, np.uint8)
        markers[markersFG > 0] = 200
        markers[markersBG > 0] = 127
        markers = markers.astype(np.int32)
        nf = NagaoFilter(21)
        bgr = nf.filter(bgr)

        cv2.watershed(bgr, markers)

        _, thresh = cv2.threshold(markers.astype(np.uint8), 150, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(thresh, mask)

    @staticmethod
    def compute_variance(bgr, window_size=11):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64)
        average = cv2.blur(gray, (window_size, window_size))

        diff = gray - average
        squared = diff * diff

        sum_averaged = cv2.blur(squared, (window_size, window_size))
        return sum_averaged
