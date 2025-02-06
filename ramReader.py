import cv2
import numpy as np

class ramReader():
    def __init__(self,env):
        self.last = 0
        self.step = 0
        self.env = env

    def reset(self):
        self.last = 0
        self.step = 0

    def getBackground(self):
        env = self.env
        ad1 = 0x0500
        ad2 = 0x05d0
        tab = [
            np.concatenate([
                env.ram[addr + j * 16:addr + (j + 1) * 16]
                for addr in (ad1, ad2, ad1, ad2)
            ])
            for j in range(13)
        ]

        if env.ram[0x071c] < self.last:
            self.step += 1
        self.last = env.ram[0x071c]

        M = -np.float32([
            [-1, 0, env.ram[0x071c] + 256*(self.step%2)],
            [0, -1, 0]
        ])

        img = np.array(tab)
        img = cv2.inRange(img, 1, 255);
        img =  cv2.bitwise_not(img)
        img = cv2.resize(img, (1024, 240), interpolation=cv2.INTER_NEAREST)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        img = img[0:240,0:352]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA )

    def getEntities(self):
        img = np.zeros((240,352,1),dtype=np.uint8)
        for i in range(1,21):
            state,xs,ys,xe,ye = self.getEntity(i)
            if state != 0x0:
                cv2.rectangle(img, (xs,ys), (xe,ye), (1), -1)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA )

    def getEntity(self,i):
        env = self.env

        state = env.ram[0x0e+i]

        xs = env.ram[0x04ac + i * 4 + 0] % 512
        ys = env.ram[0x04ac + i * 4 + 1] % 512 - 12
        xe = env.ram[0x04ac + i * 4 + 2] % 512
        ye = env.ram[0x04ac + i * 4 + 3] % 512 - 2

        return [state,xs,ys,xe,ye]
