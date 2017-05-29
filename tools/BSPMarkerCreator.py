import cv2
from tools.FileManager import FileManager
import re, os
import numpy as np
from core.MarkersSelector import MarkersSelector

class BSPMarkerCreator:

    @staticmethod
    def create_markers(mask, input_rgb_labels_folder="images/labels", output_markers_path="outputs/bsp_markers", foreground_channel=0, skeletonize=True):
        reg = r'\w+\.(jpg|jpeg|png)'
        labels = sorted([f for f in os.listdir(input_rgb_labels_folder) if re.match(reg, f.lower())])

        for lab_name in labels:
            label = FileManager.LoadImage(lab_name, input_rgb_labels_folder)
            marker = BSPMarkerCreator.create_marker(label, mask, foreground_channel, skeletonize=skeletonize)
            FileManager.SaveImage(marker, lab_name, output_markers_path)

    @staticmethod
    def create_marker(rgb_label, mask, foreground_channel=0, skeletonize=True):
        foreground_channel = int(abs(foreground_channel))
        foreground_channel = foreground_channel if foreground_channel < 3 else 2

        b, g, r = cv2.split(rgb_label)
        channels = [b, g, r]

        foreground = channels[foreground_channel]
        channels.pop(foreground_channel)
        background = cv2.bitwise_or(channels[0], channels[1])

        shape = rgb_label.shape
        ker_size = int(shape[1] * 0.01)
        ker_size = ker_size if ker_size % 2 != 0 else ker_size + 1
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ker_size, ker_size))
        cforeground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, ker)
        cbackground = cv2.morphologyEx(background, cv2.MORPH_CLOSE, ker)

        #lines = BSPMarkerCreator.draw_lines(shape)

        #red = cv2.bitwise_and(cforeground, lines, mask=foregroud)
        #blue = cv2.bitwise_and(cbackground, lines, mask=background)

        markers = np.ones(shape, np.uint8)
        markers *= 128

        if skeletonize:
            cforeground = MarkersSelector.skeletonization(cforeground)
            cbackground = MarkersSelector.skeletonization(cbackground)
        else:
            kerode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cforeground = cv2.erode(cforeground, kerode)
            cbackground = cv2.erode(cbackground, kerode)

        inv_mask = cv2.bitwise_not(mask)
        inv_mask = cv2.dilate(inv_mask, ker, iterations=2)
        ellipse = cv2.bitwise_and(mask, inv_mask)

        markers[cforeground > 0] = (0, 0, 255)
        markers[cbackground > 0] = (255, 0, 0)
        markers[(ellipse > 0) & (foreground > 0)] = (0, 0, 255)
        markers[(ellipse > 0) & (background > 0)] = (255, 0, 0)
        markers[mask == 0] = (255, 0, 0)
        return markers

    @staticmethod
    def draw_lines(shape):
        lines = np.zeros((shape[0], shape[1]), np.uint8)
        nb_line_ver = shape[1] / 10
        nb_line_hor = shape[0] / 10
        step_ver = shape[1] / nb_line_ver
        step_hor = shape[0] / nb_line_hor
        for i in range(nb_line_ver):
            cv2.line(lines, (i * step_ver, 0), (i * step_ver, shape[1]), (255, 255, 255), 2)
        for j in range(nb_line_ver):
            cv2.line(lines, (0, j * step_hor), (shape[0], j * step_hor), (255, 255, 255), 2)
        return lines