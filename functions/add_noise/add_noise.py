import numpy as np
import random

class add_noise(object):
    # 定义添加椒盐噪声的函数
    def addSaltNoise(self, img1, snr):
        # 指定信噪比
        SNR = snr
        # 获取总共像素个数
        img = img1.copy()
        size = img.size
        # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
        noiseSize = int(size * (1 - SNR))
        # 对这些点加噪声
        for k in range(0, noiseSize):
            # 随机获取 某个点
            xi = int(np.random.uniform(0, img.shape[1]))
            xj = int(np.random.uniform(0, img.shape[0]))
            # 增加噪声
            if (int(np.random.uniform(0, 2)) == 1):
                img[xj, xi] = 255
            else:
                img[xj, xi] = 0;
        #         if img.ndim == 2:
        #             img[xj, xi] = 255
        #         elif img.ndim == 3:
        #             img[xj, xi] = 0
        return img

    # 定义添加高斯函数
    #   `GaussianNoise(gray, 0, 0.1, 8)
    def GaussianNoise(self, src, means, sigma, k):
        NoiseImg = src.copy()
        rows = NoiseImg.shape[0]
        cols = NoiseImg.shape[1]
        for i in range(rows):
            for j in range(cols):
                # for x in range(3):
                NoiseImg[i, j] = NoiseImg[i, j] + k * random.gauss(means, sigma)
                if NoiseImg[i, j] < 0:
                    NoiseImg[i, j] = 0
                elif NoiseImg[i, j] > 255:
                    NoiseImg[i, j] = 255
        return NoiseImg