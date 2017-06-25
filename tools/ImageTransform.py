import cv2
import numpy as np
import math
from tools.MaskCreator import MaskCreator


class ImageTransform:
    """Class to resample an image to ignore void pixels"""

    def __init__(self, width, height, center, radius, delta_theta_deg=1.0, delta_r=0.5):
        self.width = width if width >= 0 else -width
        self.height = height if height >= 0 else -height
        self.radius = radius if radius >= 0 else -radius
        assert len(center) == 2, "Center must be a tuple or a list of length two"
        center = tuple([int(abs(c)) for c in center])
        self.center = center
        self.delta_theta_deg = abs(delta_theta_deg)
        self.delta_r = abs(delta_r)

        self.theta = 360.

        self.map_x = None
        self.map_y = None
        self.inv_map_x = None
        self.inv_map_y = None

        self.mask = MaskCreator.create_circle_mask(self.radius * 2)

        self.init_transforms()

    def init_transforms(self):
        map_x = np.zeros((int(math.ceil(self.radius / self.delta_r)), int(math.ceil(self.theta / self.delta_theta_deg))), np.float32)
        map_y = map_x.copy()
        inv_map_x = np.zeros((self.height, self.width), np.float32)
        inv_map_y = inv_map_x.copy()

        cx = self.center[0]
        cy = self.center[1]
        rx = 0
        rth = 0
        for r in np.arange(0., self.radius, self.delta_r):
            for th in np.arange(0., self.theta, self.delta_theta_deg):
                u = cx + r * math.cos(th * (math.pi / 180.))
                v = cy + r * math.sin(th * (math.pi / 180.))
                map_x[rx, rth] = u
                map_y[rx, rth] = v

                rth += 1
            rx += 1
            rth = 0

        for y in range(inv_map_x.shape[0]):
            for x in range(inv_map_x.shape[1]):
                inv_map_y[y, x] = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / self.delta_r
                dy = (y - cy)
                dx = (x - cx)
                theta_rad = math.atan2(dy, dx)
                theta_rad = theta_rad + 2 * math.pi if theta_rad < 0 else theta_rad
                theta_deg = (theta_rad * (180. / math.pi)) / self.delta_theta_deg
                inv_map_x[y, x] = theta_deg

        self.map_x = map_x
        self.map_y = map_y
        self.inv_map_x = inv_map_x
        self.inv_map_y = inv_map_y

    def torify_image(self, image, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        assert image.shape[1] == self.width and image.shape[0] == self.height, "This Image transform instance cannot handle transformation for image of shape (%d, %d)" % tuple(image.shape[:2])

        dst = cv2.remap(image, self.map_x, self.map_y, interpolation=interpolation, borderMode=border_mode, borderValue=border_value)
        return dst

    def untorify_image(self, image, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_DEFAULT):
        dst = cv2.remap(image, self.inv_map_x, self.inv_map_y, interpolation=interpolation, borderMode=border_mode)
        return cv2.bitwise_and(dst, dst, mask=self.mask)
