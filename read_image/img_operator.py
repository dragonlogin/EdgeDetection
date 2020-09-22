import cv2
import scipy.io as scio
import numpy as np


class img_operator:
    def __init__(self):
        self.path = '../image/1.png'

    def print(self, name, gray):
        print(name + '=\n{}\n'.format(gray))

    def qu(self, gray, rc):
        return gray[rc[0]:rc[1] + 1, rc[2]:rc[3] + 1]

    def readImg(self, path):
        gray = cv2.imread(path, 0)
        gray = np.array(gray)
        return gray

    def showImg(self, title, gray):
        cv2.imshow(title, gray)

    def showMat(self, mat):
        print('mat=\n{}'.format(mat))

    # 截取小片段，测试用了
    def returnMat(self):
        gray = self.readImg(self.path)
        return self.qu(gray, 50, 58, 205, 213)

    def readGroundTruth(self, truthPath='../image/8068.mat'):
        # path = './Mat/8068.mat'
        groundTrhth = truthPath
        matdata = scio.loadmat(groundTrhth)
        # matgray = matdata['groundTruth'][0][0][0][0][1] * 255
        # cv2.imshow('groundTruth', matgray)
        # cv2.waitKey(0)
        matgray = matdata['groundTruth'][0][0][0][0][1]

        for k in range(1, 6):
            for i in range(len(matgray)):
                for j in range(len(matgray[0])):
                    if (matgray[i][j] != 1):
                        matgray[i][j] += matdata['groundTruth'][0][k][0][0][1][i][j]

        matgray = matgray * 255
        return matgray

    def returnGroundTruthMat(self, truthPath='../image/8068.mat'):
        groundTrhth = truthPath
        matdata = scio.loadmat(groundTrhth)
        # print(matdata['groundTruth'])
        matgray = matdata['groundTruth'][0][0][0][0][1] * 255  # 修改第二个0
        # print('matgray=\n', matgray)
        return matgray

    def showAllGroundTruth(self, path):
        from pylab import plt, imshow, subplot
        matdata = scio.loadmat(path)
        n = len(matdata['groundTruth'][0])
        mp = {}

        for i in range(n):
            mp[i] = matdata['groundTruth'][0][i][0][0][1] * 255
            cv2.imshow('truth' + str(i), mp[i])

        # htitch = np.hstack(list(mp[i] for i in mp.keys()))
        # cv2.imshow("test1", htitch)
        # cv2.waitKey(0)

    def test(self):
        # -*- coding: utf-8 -*-
        # 利用 np.hstack、np.vstack实现一幅图像中显示多幅图片
        """
        created on Thursday June 14 17:05 2018
        @author: Jerry

        """
        import cv2
        # from pylab import *

        img1 = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
        img2 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread('lena.jpg', cv2.IMREAD_UNCHANGED)
        img4 = cv2.imread('lena.jpg')

        htitch = np.hstack((img1, img3, img4))
        vtitch = np.vstack((img1, img3))
        cv2.imshow("test1", htitch)
        cv2.imshow("test2", vtitch)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
# io = ImgOperator()
# io.showAllGroundTruth()
