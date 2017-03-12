import numpy as np
import math

class OpticalRectifier:

    def __init__(self, tableSrc, imgViewAngle, brightnessScaleFactor, brightnessScaleOffset):
        self.tableSrc = []
        sum = 0.0
        for dst in tableSrc:
            sum += dst
            self.tableSrc.append(float(sum))
        self.imgViewAngle = imgViewAngle
        self.brightnessScaleFactor = brightnessScaleFactor
        self.brightnessScaleOffset = brightnessScaleOffset

    def rectifyImage(self, img):
        height, width, channels = img.shape
        rI = width / 2.0

        xCenter = width / 2
        yCenter = height / 2

        tableCorr = self.getRectifiedCalibTable(rI)

        f = np.poly1d(np.polyfit(tableCorr, np.asarray(self.tableSrc) * rI / self.tableSrc[-1], 2))

        #with open("coucou.txt", "w") as fer:
        #    for i in range(int(rI)):
        #        fer.write("%f\n"% (f(i)))

        result = np.zeros(img.shape, np.uint8)
        for yCor in range(height):
            for xCor in range(width):
                dx = xCor - xCenter
                dy = yCor - yCenter
                rCor = math.sqrt(dx * dx + dy * dy)
                angle = dx/rCor if rCor != 0 else 0
                alpha = math.asin(angle)
                if yCor > yCenter:
                    alpha = math.pi - alpha
                rSrc = f(rCor)

                if(rSrc) < rI:
                    xSrc = int(xCenter + rSrc * math.sin(alpha))
                    ySrc = int(yCenter - rSrc * math.cos(alpha))

                    rgb = img[xSrc, ySrc]
                    for channel in range(channels):
                        result[xCor, yCor][channel] = min(int(rgb[channel] * self.brightnessScaleFactor + self.brightnessScaleOffset), 255)

        return result

    def getRectifiedCalibTable(self, rI):
        length = len(self.tableSrc)
        assert length > 0
        return np.asarray([(rI / length) * (i + 1) for i in range(length)])
