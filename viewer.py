import time
import cv2
import numpy as np
import torch


def drawLine(frame, start, resolution, coefx, coefy):
    x = start.x
    y = start.y

    if (x >= resolution.x or x < 0 or y >= resolution.y or y < 0):
        return Sensor(Vector(0,0),Vector(0,0))

    while frame[y][x][0] == 255:
        x += coefx
        y += coefy

        willbreak = False

        if (x >= resolution.x or x < 0):
            x -= coefx
            willbreak = True

        if (y >= resolution.y or y < 0):
            y -= coefy
            willbreak = True

        if (willbreak):
            break

    end = Vector(x, y)

    return Sensor(start, end)


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toList(self):
        return [self.x, self.y]


class Sensor:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def distance(self):
        a = self.p1.x - self.p2.x
        b = self.p1.y - self.p2.y
        return np.sqrt((a**2) + (b**2))


class Viewer:
    def __init__(self, title, resolution):
        self.frame = []
        self.sensors = []
        self.resolution = resolution
        self.codec = cv2.VideoWriter_fourcc(*"XVID")
        self.fps = 60.0
        self.title = title

        self.sensorsN = 8

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.resolution.x, self.resolution.y)

    def _updateSensors(self,player):
        _,xLeft,yTop,xRight,yBottom = player

        xMiddle =  (xRight+xLeft)//2
        yMiddle =  (yTop+yBottom)//2

        self.sensors = []

        # coin
        self.sensors.append( drawLine(self.frame, Vector(xLeft,yTop), self.resolution, -1, -1) )
        self.sensors.append( drawLine(self.frame, Vector(xRight,yTop), self.resolution, 1, -1) )
        self.sensors.append( drawLine(self.frame, Vector(xLeft,yBottom), self.resolution, -1, 1) )
        self.sensors.append( drawLine(self.frame, Vector(xRight,yBottom), self.resolution, 1, 1) )

        # cotés
        self.sensors.append( drawLine(self.frame, Vector(xLeft,yMiddle), self.resolution, -1, 0) )
        self.sensors.append( drawLine(self.frame, Vector(xRight,yMiddle), self.resolution, 1, 0) )

        # haut/base
        self.sensors.append(drawLine(self.frame, Vector(xMiddle, yTop), self.resolution, 0, -1))
        self.sensors.append(drawLine(self.frame, Vector(xMiddle, yBottom), self.resolution, 0, 1))

    def getTorchSensors(self):
        a = [x.distance() for x in self.sensors]
        b = torch.FloatTensor(a)
        return b

    def _updateFrame(self,img):
        #img = cv2.colorChange(img,cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.resolution.toList(), interpolation=cv2.INTER_LINEAR)
        self.frame = img

    def update(self,img,player):
        self._updateFrame(img)
        self._updateSensors(player)

    def render(self):
        img = self.frame
        for s in self.sensors:
            cv2.line(img, s.p1.toList(), s.p2.toList(), (255, 0, 0), 1)

        cv2.imshow(self.title, img)

    def isOpened(self):
        if cv2.waitKey(1) == ord('q'):
            return False
        return True

    def destroy(self):
        cv2.destroyAllWindows()