import cv2
import os


class FileManager:

    @staticmethod
    def LoadImage(filename, path="images/", flags=cv2.IMREAD_COLOR):
        if(os.path.exists(path)):
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath, flags)
            return img
        return None

    @staticmethod
    def SaveImage(img, filename, output_folder="outputs/"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, img)