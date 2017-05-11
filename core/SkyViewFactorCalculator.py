import cv2
import math
import numpy as np
from tools.FileManager import FileManager


class SkyViewFactorCalculator:

    @staticmethod
    def compute_factor(binary_mask, number_of_steps=200, center=(720, 720), radius=720):
        assert len(center) == 2, "SVFCalculator: Center must be a tuple or a list of length 2"
        center = tuple([int(abs(c)) for c in center])
        radius = abs(radius)
        number_of_steps = abs(number_of_steps) if number_of_steps != 0 else 10
        assert number_of_steps < radius, "SVFCalculator: The number of steps cannot be greater than the radius in pixels, pixels cannot be divided in smaller pieces sorry..."

        thickness = int(math.floor(radius / number_of_steps))
        first_thickness = radius % number_of_steps

        if first_thickness > thickness:
            add_steps = int(math.floor(first_thickness / thickness))
            number_of_steps += add_steps
            first_thickness %= add_steps
            print "SVFCalculator: %d steps have been added to fit more closely the discretization of the radius" % add_steps

        current_radius = first_thickness

        factor = 0.
        constant = (1.0 / (2.0 * math.pi)) * math.sin(math.pi / (2.0 * number_of_steps))

        tmp_mask = None

        for i in range(0, number_of_steps):
            mask = np.zeros(binary_mask.shape, np.uint8)
            if i == 0:
                cv2.circle(mask, center, first_thickness + int(thickness / 2), (255, 255, 255), -1, lineType=8)
            else:
                cv2.circle(mask, center, current_radius, (255, 255, 255), thickness, lineType=8)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(tmp_mask))

            tmp_mask = mask.copy()
            mask_count = cv2.countNonZero(mask)
            included = cv2.bitwise_and(binary_mask, mask)
            pixels = cv2.countNonZero(included)

            mask_count = 1 if mask_count == 0 else mask_count

            alpha_i = (pixels / float(mask_count)) * 2 * math.pi

            sin_part = (math.pi * (2.0 * (i + 1) - 1)) / (2.0 * number_of_steps)

            factor += math.sin(sin_part) * alpha_i
            current_radius += thickness

        factor *= constant
        return factor

    @staticmethod
    def compute_factor_bgr_labels(bgr, number_of_steps=100, center=(720, 720), radius=720):
        b, g, r = cv2.split(bgr)

        chans = [b, g, r]
        factors = []
        for c in chans:
            factors.append(SkyViewFactorCalculator.compute_factor(c, number_of_steps=number_of_steps,
                                                                  center=center, radius=radius))
        return factors
