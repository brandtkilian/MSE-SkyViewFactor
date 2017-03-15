import numpy as np
import math
from tools.FuncTimeProfiler import profile


class OpticalRectifier:

    def __init__(self, tableSrc, imgViewAngle, imgWidth, imgHeight):
        self.tableSrc = []
        sum = 0.0
        for dst in tableSrc:
            sum += dst
            self.tableSrc.append(float(sum))
        self.imgViewAngle = imgViewAngle
        self.mapping = None
        self.InitializeCoordsTransform(imgWidth, imgHeight)


    @profile
    def rectifyImage(self, img):
        result = img[self.mapping[..., 1], self.mapping[..., 0]]
        return result

    @profile
    def InitializeCoordsTransform(self, width, height):
        if self.mapping is None:
            self.mapping = np.zeros((height, width, 2), np.int32)
            self.mask = np.ones((height, width, 1), np.uint8)

            rI = width / 2.0
            xCenter = width / 2
            yCenter = height / 2

            tableCorr = self.getRectifiedCalibTable(rI)
            f = np.poly1d(np.polyfit(tableCorr, np.asarray(self.tableSrc) * rI / self.tableSrc[-1], 2))

            for yCor in range(height):
                for xCor in range(width):
                    dx = xCor - xCenter
                    dy = yCor - yCenter
                    rCor = math.sqrt(dx * dx + dy * dy)
                    angle = dx / rCor if rCor != 0 else 0
                    alpha = math.asin(angle)
                    if yCor > yCenter:
                        alpha = math.pi - alpha
                    rSrc = f(rCor)

                    if (rSrc) < rI:
                        xSrc = int(xCenter + rSrc * math.sin(alpha))
                        ySrc = int(yCenter - rSrc * math.cos(alpha))
                        self.mapping[yCor, xCor, 0] = xSrc
                        self.mapping[yCor, xCor, 1] = ySrc


    def getRectifiedCalibTable(self, rI):
        length = len(self.tableSrc)
        assert length > 0
        return np.asarray([(rI / length) * (i + 1) for i in range(length)])
