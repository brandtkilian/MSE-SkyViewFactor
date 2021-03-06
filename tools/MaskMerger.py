import cv2
import glob
from tools.FileManager import FileManager


class MasksMerger():
    """Static class to merge masks created with IST and resolve conflicts"""

    @staticmethod
    def merge_from_sky_and_build(buildPath, skyPath, mask, output_dir="outputs/"):
        files1 = [f for f in glob.iglob("%s/*.png" % buildPath)]
        files2 = [f for f in glob.iglob("%s/*.png" % skyPath)]

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
                    bc, gc, rc = cv2.split(crop)
                    nb = cv2.countNonZero(bc)
                    ng = cv2.countNonZero(gc)
                    nr = cv2.countNonZero(rc)
                    counts = (nb, ng, nr)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    winner = sorted(enumerate(counts), key=lambda k: k[1], reverse=True)
                    idx, count = winner[0]
                    color = colors[idx]
                    cv2.drawContours(merged, [c], 0, color, -1)

            b, g, r = cv2.split(merged)
            conflicts = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_and(r, b),
                                       cv2.bitwise_and(r, g)), cv2.bitwise_and(b, r))
            idx = conflicts > 0
            merged[idx] = (255, 255, 255)
            merged = cv2.bitwise_and(merged, merged, None, mask)
            FileManager.SaveImage(merged, files1[i], output_dir)

    @staticmethod
    def merge_masks_from_all(sky_path, veg_path, built_path, mask, output_dir="outputs/"):
        sky_imgs = sorted([f for f in glob.iglob("%s/*.png" % sky_path)])
        veg_imgs = sorted([f for f in glob.iglob("%s/*.png" % veg_path)])
        built_imgs = sorted([f for f in glob.iglob("%s/*.png" % built_path)])

        assert len(sky_imgs) == len(veg_imgs) and len(veg_imgs) == len(built_imgs), "Numbers of images in the three given pathes aren't equals"

        for sky, veg, built in zip(sky_imgs, veg_imgs, built_imgs):
            b = FileManager.LoadImage(sky, sky_path, cv2.IMREAD_GRAYSCALE)
            g = FileManager.LoadImage(veg, veg_path, cv2.IMREAD_GRAYSCALE)
            #r = cv2.bitwise_not(cv2.bitwise_or(b, g))
            r = FileManager.LoadImage(built, built_path, cv2.IMREAD_GRAYSCALE)
            merged = cv2.merge((b, g, r))
            merged = cv2.bitwise_and(merged, merged, None, mask)
            a = cv2.bitwise_or(b, g)
            a = cv2.bitwise_or(a, r)
            a = cv2.bitwise_or(a, cv2.bitwise_not(mask))
            a = cv2.bitwise_not(a)
            n = cv2.countNonZero(a)
            if n < 100:
                FileManager.SaveImage(merged, sky, output_dir)
