import cv2
import numpy as np


class MaskCreator():

    @staticmethod
    def create_circle_mask(diameter):
        mask = np.zeros((diameter, diameter), np.uint8)
        cv2.circle(mask, (diameter / 2, diameter / 2), diameter / 2, (255, 255, 255), -1)
        return mask

    @staticmethod
    def create_circle_maskwh(width, height, radius=-1):
        mask = np.zeros((width, height), np.uint8)
        radius = width / 2 if radius < 0 else radius
        cv2.circle(mask, (width / 2, height / 2), radius, (255, 255, 255), -1)
        return mask
