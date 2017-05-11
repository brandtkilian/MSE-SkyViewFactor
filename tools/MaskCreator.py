import cv2
import numpy as np

class MaskCreator():

    @staticmethod
    def create_circle_mask(diameter):
        mask = np.zeros((diameter, diameter), np.uint8)
        cv2.circle(mask, (diameter / 2, diameter / 2), diameter / 2, (255, 255, 255), -1)
        return mask
