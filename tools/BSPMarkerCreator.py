import cv2
from tools.FileManager import FileManager
import re, os
import numpy as np

class BSPMarkerCreator:

    @staticmethod
    def create_markers(input_rgb_labels_folder="images/labels", output_markers_path="outputs/bsp_markers", foreground_channel=0):
        reg = r'\w+\.(jpg|jpeg|png)'
        labels = sorted([f for f in os.listdir(input_rgb_labels_folder) if re.match(reg, f.lower())])

        for lab_name in labels:
            label = FileManager.LoadImage(lab_name, input_rgb_labels_folder)
            marker = BSPMarkerCreator.create_marker(label, foreground_channel)
            FileManager.SaveImage(marker, lab_name, output_markers_path)

    @staticmethod
    def create_marker(rgb_label, foreground_channel=0):
        foreground_channel = int(abs(foreground_channel))
        foreground_channel = foreground_channel if foreground_channel < 3 else 2

        b, g, r = cv2.split(rgb_label)
        channels = [b, g, r]

        foregroud = channels[foreground_channel]
        channels.pop(foreground_channel)
        background = cv2.bitwise_or(channels[0], channels[1])

        shape = rgb_label.shape
        ker_size = int(shape[1] * 0.01)
        ker_size = ker_size if ker_size % 2 != 0 else ker_size + 1
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ker_size, ker_size))
        cforeground = cv2.morphologyEx(foregroud, cv2.MORPH_CLOSE, ker)
        cbackground = cv2.morphologyEx(background, cv2.MORPH_CLOSE, ker)

        cforeground = cv2.erode(cforeground, ker)
        cbackground = cv2.erode(cbackground, ker)

        lines = np.zeros((shape[0], shape[1]), np.uint8)
        nb_line_ver = shape[1] / 20
        nb_line_hor = shape[0] / 20
        step_ver = shape[1] / nb_line_ver
        step_hor = shape[0] / nb_line_hor
        for i in range(nb_line_ver):
            cv2.line(lines, (i * step_ver, 0), (i * step_ver, shape[1]), (255, 255, 255), 2)
        for j in range(nb_line_ver):
            cv2.line(lines, (0, j * step_hor), (shape[0], j * step_hor), (255, 255, 255), 2)

        red = cv2.bitwise_and(cforeground, lines, mask=foregroud)
        blue = cv2.bitwise_and(cbackground, lines, mask=background)

        markers = np.ones(shape, np.uint8)
        markers *= 128

        markers[red > 0] = (0, 0, 255)
        markers[blue > 0] = (255, 0, 0)
        return markers