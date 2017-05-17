import cv2
import math
import numpy as np
from tools.FileManager import FileManager
from sklearn.metrics import mean_squared_error


class SkyViewFactorCalculator:

    @staticmethod
    def compute_factor(binary_mask, number_of_steps=120, center=(720, 720), radius=720):
        assert len(center) == 2, "SVFCalculator: Center must be a tuple or a list of length 2"
        center = tuple([int(abs(c)) for c in center])
        radius = abs(radius)
        assert center[0] + radius <= binary_mask.shape[1] and center[0] - radius >= 0 and center[1] + radius <= binary_mask.shape[0] and center[1] - radius >= 0, "Radius and center values incoherent regarding the input mask size..."
        number_of_steps = abs(number_of_steps) if number_of_steps != 0 else 10
        assert number_of_steps <= radius, "SVFCalculator: The number of steps cannot be greater than the radius in pixels, pixels cannot be divided in smaller pieces sorry..."

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
    def compute_factor_bgr_labels(bgr, number_of_steps=120, center=(720, 720), radius=720):
        b, g, r = cv2.split(bgr)

        chans = [b, g, r]
        factors = []
        for c in chans:
            factors.append(SkyViewFactorCalculator.compute_factor(c, number_of_steps=number_of_steps,
                                                                  center=center, radius=radius))
        return factors

    @staticmethod
    def compute_factor_annotated_label(annotated, class_label=1, number_of_steps=120, center=(720, 720), radius=720):
        class_label = int(class_label)
        idx = annotated == class_label
        mask = np.zeros(annotated.shape, np.uint8)
        mask[idx] = 255

        return SkyViewFactorCalculator.compute_factor(mask, number_of_steps=number_of_steps,
                                                      center=center, radius=radius)

    @staticmethod
    def compute_factor_annotated_labels(annotated, class_labels=[1, 2, 3], number_of_steps=120, center=(720, 720), radius=720):
        assert isinstance(class_labels, (list, tuple))
        factors = []
        for c in class_labels:
            factors.append(SkyViewFactorCalculator.compute_factor_annotated_label(annotated, c,
                                                                                  number_of_steps=number_of_steps,
                                                                                  center=center, radius=radius))
        return factors

    @staticmethod
    def compute_mean_square_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def compute_sky_angle_estimation(binary_mask, center, radius_low, radius_top, epsilon=1e-2, sky_view_factor=-1,
                                     number_of_steps=120, center_factor=(720, 720), radius_factor=720):
        range_radius = abs(radius_top - radius_low)
        first_radius = int(math.ceil(radius_low + 0.25 * range_radius))
        second_radius = int(math.ceil(radius_low + 0.75 * range_radius))

        if sky_view_factor < 0:
            sky_view_factor = SkyViewFactorCalculator.compute_factor(binary_mask, number_of_steps=number_of_steps,
                                                                     center=center_factor, radius=radius_factor)

        first_mask = np.zeros(binary_mask.shape, np.uint8)
        second_mask = np.zeros(binary_mask.shape, np.uint8)
        cv2.circle(first_mask, center, first_radius, (255, 255, 255), -1, lineType=8)
        cv2.circle(second_mask, center, second_radius, (255, 255, 255), -1, lineType=8)

        first_overlap = cv2.bitwise_and(binary_mask, first_mask)
        second_overlap = cv2.bitwise_and(binary_mask, second_mask)


        first_svf = SkyViewFactorCalculator.compute_factor(first_overlap, number_of_steps=number_of_steps,
                                                           center=center_factor, radius=radius_factor)
        second_svf = SkyViewFactorCalculator.compute_factor(second_overlap, number_of_steps=number_of_steps,
                                                            center=center_factor, radius=radius_factor)

        expected = 0.5 * sky_view_factor
        first_diff = abs(expected - first_svf)
        second_diff = abs(expected - second_svf)

        radius_angle_factor = 90.0 / radius_factor
        if first_diff < epsilon:
            return first_radius * radius_angle_factor
        elif second_diff < epsilon:
            return second_radius * radius_angle_factor

        if first_diff < second_diff:
            radius_low = int(math.ceil(first_radius - 0.25 * range_radius))
            radius_top = int(math.ceil(first_radius + 0.25 * range_radius))
        else:
            radius_low = int(math.ceil(second_radius - 0.25 * range_radius))
            radius_top = int(math.ceil(second_radius + 0.25 * range_radius))

        return SkyViewFactorCalculator.compute_sky_angle_estimation(binary_mask, center, radius_low, radius_top,
                                                                    sky_view_factor=sky_view_factor,
                                                                    number_of_steps=number_of_steps,
                                                                    center_factor=center_factor,
                                                                    radius_factor=radius_factor)


