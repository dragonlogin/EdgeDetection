import numpy as np
import cv2
import random
from collections import deque


class enhance_unit(object):
    def __init__(self):
        filePath = '../image/lena.jpg'
        self.gray = cv2.imread(filePath, 0)
        cv2.imshow('origin', self.gray)
        # self.num = 8
        self.noiseSho = 7

        # self.gray = cv2.imread(filePath, 0)

    '''
    
    # 根据中心点像素和与之的相对位置求八邻域点内的坐标
    # 参数：中心像素点坐标，相对位置（从左上为0顺指针标记）,邻域个数
    # 得到：邻域点像素
    
    '''

    def findIndex(self, i, j, x, num=8):
        ii, jj = 0, 0

        if num == 8:
            if ((x == 0) | (x == 1) | (x == 2)):
                ii = i - 1
            if ((x == 4) | (x == 5) | (x == 6)):
                ii = i + 1
            if ((x == 3) | (x == 7)):
                ii = i

            if ((x == 0) | (x == 6) | (x == 7)):
                jj = j - 1
            if ((x == 2) | (x == 3) | (x == 4)):
                jj = j + 1
            if ((x == 1) | (x == 5)):
                jj = j
        if num == 16:
            if ((x == 0) | (x == 1) | (x == 2) | (x == 3) | (x == 4)):
                ii = i - 2
            if ((x == 5) | (x == 15)):
                ii = i - 1
            if ((x == 6) | (x == 14)):
                ii = i
            if ((x == 7) | (x == 13)):
                ii = i + 1
            if ((x == 8) | (x == 9) | (x == 10) | (x == 11) | (x == 12)):
                ii = i + 2

            if ((x == 0) | (x == 15) | (x == 14) | (x == 13) | (x == 12)):
                jj = j - 2
            if ((x == 1) | (x == 11)):
                jj = j - 1
            if ((x == 2) | (x == 10)):
                jj = j
            if ((x == 3) | (x == 9)):
                jj = j + 1
            if ((x == 4) | (x == 5) | (x == 6) | (x == 7) | (x == 8)):
                jj = j + 2

        return ii, jj

    '''
    # 找八邻域点，组成列表
    # 参数：灰度矩阵，像素点坐标,邻域个数
    # 得到：邻域点(值)列表
    '''

    def point_list(self, mat, i, j, num=8):
        h, w = mat.shape

        if num == 4:
            lists = []  # 四邻域灰度值从北，东南西 顺时针排列
            addr = []  # 在mat中的坐标
            addr1 = []
            addr1.extend(((i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)))

            for (n, m) in addr1:
                # print((n,m))
                # 判断是否出了边界
                if ((n < 0) | (n >= h)) == 0:
                    if ((m < 0) | (m >= w)) == 0:
                        lists.append(mat[n, m]);
                        addr.append((n, m))

            # lists.extend([mat[i - 1, j],mat[i, j + 1],mat[i + 1, j],mat[i, j-1]]);
            # addr.extend(( (i - 1, j),  (i, j + 1), (i + 1, j), (i, j - 1)))

        if num == 9:
            lists = []  # 八邻域灰度值从左上开始沿顺时针方向排列
            addr = []  # 在mat中的坐标
            addr1 = []

            addr1.extend(
                ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1),
                 (i, j - 1), (i, j)))

            for (n, m) in addr1:
                if ((n < 0) | (n >= h)) == 0:
                    if ((m < 0) | (m >= w)) == 0:
                        lists.append(int(mat[n, m]))
                        addr.append((n, m))

        if num == 8:
            lists = []  # 八邻域灰度值从左上开始沿顺时针方向排列
            addr = []  # 在mat中的坐标
            addr1 = []

            addr1.extend(
                ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1),
                 (i, j - 1)))

            for (n, m) in addr1:
                # print((n,m))
                if ((n < 0) | (n >= h)) == 0:
                    if ((m < 0) | (m >= w)) == 0:
                        lists.append(int(mat[n, m]))
                        addr.append((n, m))

        if num == 16:
            lists = []  # 八邻域灰度值从左上开始沿顺时针方向排列
            addr = []
            addr1 = []

            addr1.extend(((i - 2, j - 2), (i - 2, j - 1), (i - 2, j), (i - 2, j + 1), (i - 2, j + 2), (i - 1, j + 2),
                          (i, j + 2), (i + 1, j + 2), (i + 2, j + 2), (i + 2, j + 1), (i + 2, j), (i + 2, j - 1),
                          (i + 2, j - 2), (i + 1, j - 2), (i, j - 2), (i - 1, j - 2)))

            for (n, m) in addr1:
                # print((n,m))
                if ((n < 0) | (n >= h)) == 0:
                    if ((m < 0) | (m >= w)) == 0:
                        lists.append(mat[n, m]);
                        addr.append((n, m))

        return lists, addr


    def partition_list(self, lists, x, y, num=8):
        # lists, _ = self.point_list(gray, x, y, num)
        # _, index = self.lockNoise(gray, x, y, num)
        # print("\nnoiseList={}\t noiseIndex={}".format(lists, index))
        A, B = [], []
        ariseIndex, decIndex = -1, -1
        ariseMax, decMin = 0, 0
        diff = 0
        # if index == None:
        #     diff = 0
        #     return diff
        # del lists[index]
        lens = len(lists)
        for i in range(lens):
            sub = int(lists[(i + 1) % lens]) - int(lists[i])
            # print("sub={}".format(sub))
            if sub > 0 and sub > ariseMax:
                ariseMax, ariseIndex = sub, i
            elif sub < 0 and sub < decMin:
                decMin, decIndex = sub, i
        # print("av={}\tai={}".format(ariseMax, ariseIndex))
        # print("dv={}\tdi={}".format(decMin, decIndex))
        minIndex, maxIndex = min(ariseIndex, decIndex), max(ariseIndex, decIndex)
        for i in range(lens):
            if minIndex < i <= maxIndex:
                B.append(lists[i])
            else:
                A.append(lists[i])

        # print("A={}\tB={}".format(A,B))
        # np.std([1, 2, 3], ddof=1)
        # 修正一下，一个数的标准差，调用函数为 nan, 应该为 0
        # 修正一下， 空列表的平均数为 0, 直接调用函数为 nan
        # VA, VB = np.mean(A), np.mean(B)

        return A, B

    def partitionAB(self, gray, x, y, num=8):
        lists, _ = self.point_list(gray, x, y, num)
        # _, index = self.lockNoise(gray, x, y, num)
        # print("\nnoiseList={}\t noiseIndex={}".format(lists, index))
        A, B = [], []
        ariseIndex, decIndex = -1, -1
        ariseMax, decMin = 0, 0
        diff = 0
        # if index == None:
        #     diff = 0
        #     return diff
        # del lists[index]
        lens = len(lists)
        for i in range(lens):
            sub = int(lists[(i + 1) % lens]) - int(lists[i])
            # print("sub={}".format(sub))
            if sub > 0 and sub > ariseMax:
                ariseMax, ariseIndex = sub, i
            elif sub < 0 and sub < decMin:
                decMin, decIndex = sub, i
        # print("av={}\tai={}".format(ariseMax, ariseIndex))
        # print("dv={}\tdi={}".format(decMin, decIndex))
        minIndex, maxIndex = min(ariseIndex, decIndex), max(ariseIndex, decIndex)
        for i in range(lens):
            if minIndex < i <= maxIndex:
                B.append(lists[i])
            else:
                A.append(lists[i])
        # print("A={}\tB={}".format(A,B))
        # np.std([1, 2, 3], ddof=1)
        # 修正一下，一个数的标准差，调用函数为 nan, 应该为 0
        # 修正一下， 空列表的平均数为 0, 直接调用函数为 nan
        # VA, VB = np.mean(A), np.mean(B)

        return A, B

    '''
    # 定位以 (x, y) 邻域为 8 为中心的噪声点
    '''
    def lockNoise(self, mat, x, y, num=8):
        lists, _ = self.point_list(mat, x, y, num)
        noiseCost = []  # 8邻域每个点的价值

        # 单调 =》 true
        def check(left, mid, right):
            if left <= mid <= right or left >= mid >= right:
                return True
            else:
                return False

        for i in range(num):  # [0,7]
            lv, mv, rv = int(lists[i - 1]), int(lists[i]), int(lists[(i + 1) % num])

            if check(lv, mv, rv):
                cost = min(abs(lv - mv), abs(mv - rv))
                noiseCost.append(cost)
                # print("{}, {}, {} = a".format(lv, mv, rv))
            else:
                cost = abs(lv - mv) + abs(mv - rv) - abs(lv - rv)
                noiseCost.append(cost)
                # print("{}, {}, {} = b".format(lv, mv, rv))

        index, maxCost = 0, noiseCost[0]

        for i in range(1, num):
            if noiseCost[i] > maxCost:
                index, maxCost = i, noiseCost[i]
        # maxCost == 0  表示全部相等

        if maxCost == 0:
            return 0, None

        return maxCost, index

    # 分区
    def partitionAB_getdiff(self, gray, x, y, num=8):
        lists, _ = self.point_list(gray, x, y, num)
        _, index = self.lockNoise(gray, x, y, num)
        # print("\nnoiseList={}\t noiseIndex={}".format(lists, index))
        A, B = [], []
        ariseIndex, decIndex = -1, -1
        ariseMax, decMin = 0, 0
        diff = 0
        if index == None:
            diff = 0
            return diff
        del lists[index]
        lens = len(lists)
        for i in range(lens):
            sub = int(lists[(i + 1) % lens]) - int(lists[i])
            # print("sub={}".format(sub))
            if sub > 0 and sub > ariseMax:
                ariseMax, ariseIndex = sub, i
            elif sub < 0 and sub < decMin:
                decMin, decIndex = sub, i
        # print("av={}\tai={}".format(ariseMax, ariseIndex))
        # print("dv={}\tdi={}".format(decMin, decIndex))
        minIndex, maxIndex = min(ariseIndex, decIndex), max(ariseIndex, decIndex)
        for i in range(lens):
            if minIndex < i <= maxIndex:
                B.append(lists[i])
            else:
                A.append(lists[i])
        # print("A={}\tB={}".format(A,B))
        # np.std([1, 2, 3], ddof=1)
        # 修正一下，一个数的标准差，调用函数为 nan, 应该为 0
        # 修正一下， 空列表的平均数为 0, 直接调用函数为 nan
        # VA, VB = np.mean(A), np.mean(B)
        if len(A) == 0:
            VA = 0
        else:
            VA = np.mean(A)
        if len(B) == 0:
            VB = 0
        else:
            VB = np.mean(B)
        if len(A) == 0 or len(A) == 1:
            EA = 0
        else:
            EA = np.std(A, ddof=1)

        if len(B) == 0 or len(B) == 1:
            EB = 0
        else:
            EB = np.std(B, ddof=1)

        # print("VA={}\tEA={}".format(VA, EA))
        # print("VB={}\tEB={}".format(VB, EB))
        subV, sumE = VA - VB, EA + EB
        if abs(subV) <= abs(sumE):
            diff = 0
        else:
            diff = abs(subV) - abs(sumE)
        # print("diff={}\n".format(diff))
        return diff
        # print(lists, '\t', index)



    # 为全图 定位噪声
    # 返回 mark 每个像素成为噪声的可能性
    def lockFullNoise(self, gray, num=8):
        h, w = gray.shape
        # print("height = {}\t width = {}".format(height, width))
        mark = np.zeros(gray.shape, np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                maxCost, index = self.lockNoise(gray, i, j, num)
                # 表示同一个区域， 没有噪声点
                if index == None:
                    continue
                posX, posY = self.findIndex(i, j, index, num)
                mark[posX][posY] += maxCost

        noiseList = []
        maxValue = 0
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                maxValue = max(maxValue, mark[i, j])
                noiseList.append(int(mark[i, j]))

        Tools().drawHist(noiseList, int(maxValue), 'noiseValue', 'number')
        noiseSho = np.mean(noiseList) + np.std(noiseList, ddof=1)
        print('noiseSho={}'.format(self.noiseSho))
        return mark, noiseSho

    # # 八邻域非重复中值修复噪声点
    # # x, y噪声中心点
    # # num 3x3矩阵
    # def fixNoise(self, x, y, num=8):
    #     lists, _ = self.point_list(gray, x, y, num)
    #     s = set(lists)
    #     lists = list(s)
    #     lists.sort()
    #     return lists[int(len(lists) / 2)]

    def fixFullNoise(self, origin, mark, sho, num=8):
        img = origin.copy()

        h, w = img.shape
        count = 0
        for i in range(h):
            for j in range(w):
                if mark[i][j] > sho:
                    count += 1
                    fixResult = self.fixNoise(i, j, 8)
                    # print("originPix={}\tfixedPix={}\t x={}\ty={}".format(img[i,j], fixResult, i, j))
                    img[i, j] = fixResult

        # print("fixNum={}".format(count))
        # print("fixFullNoise OK")
        return img



    def printEightNeighboor(self, sho):
        mark = self.lockFullPicture(self.num)
        for i in range(self.height):
            for j in range(self.width):
                if mark[i][j] >= sho:
                    print("\ni={}\tj={},\tmarkValue={}\n".format(i, j, int(mark[i][j])))
                    R = np.arange(i - 1, i + 2, 1)
                    C = np.arange(j - 1, j + 2, 1)
                    area = self.gray[R]
                    area = area[:, C]
                    print('\n', area, '\n')
        print("printEightNeighboor Ok")

    # 直接和 diffSho 比较，> 为噪声点
    def drawEdgeDetection(self, gray, sho):
        img = gray.copy()
        edgeCount = 0
        h, w = img.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                diff = self.partitionAB(gray, i, j)
                if diff > sho:
                    img[i, j] = 255
                    edgeCount += 1
                else:
                    img[i, j] = 0
        print("rate={}".format(edgeCount / (w * h)))
        Tools().drawBinPic('>', img)

    def blurAndSNR(self, origin, noiseI):
        noiseImg = noiseI.copy()
        # 不滤波
        medianResult = self.SNR(origin, noiseImg)
        self.print_SNR("没有滤波", medianResult)

        # 中值滤波

        img_medianBlur = cv2.medianBlur(noiseImg, 5)
        medianResult = self.SNR(origin, img_medianBlur)
        self.print_SNR("中值滤波", medianResult)
        # 均值滤波
        img_Blur = cv2.blur(noiseImg, (5, 5))
        blurResult = self.SNR(origin, img_Blur)
        self.print_SNR("均值滤波", blurResult)
        # 高斯滤波
        img_GaussianBlur = cv2.GaussianBlur(noiseImg, (7, 7), 0)
        GaussianResult = self.SNR(origin, img_GaussianBlur)
        self.print_SNR("高斯滤波", GaussianResult)
        # 高斯双边滤波
        img_bilateralFilter = cv2.bilateralFilter(noiseImg, 40, 75, 75)
        bilResult = self.SNR(origin, img_bilateralFilter)
        self.print_SNR("高斯双边滤波", bilResult)
        # 我的方法
        mark, noiseSho = self.lockFullNoise(noiseI)
        ourBlur = self.fixFullNoise(noiseImg, mark, noiseSho)
        ourResult = self.SNR(origin, ourBlur)
        self.print_SNR("我的滤波", ourResult)

        print("信噪比越大越好")

    def addNoise(self, img, snr):
        h = img.shape[0]
        w = img.shape[1]
        img1 = img.copy()
        sp = h * w  # 计算图像像素点个数
        NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
        for i in range(NP):
            randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
            randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
            if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
                img1[randx, randy] = 0
            else:
                img1[randx, randy] = 255
        return img1, NP

    def BFS(self, mark, x, y, diffSho, img, markNum):
        d = deque()
        d.append((x, y))
        h, w = img.shape
        dir = [-1, 0, 1, 0, -1]
        while len(d) > 0:
            size = len(d)
            for i in range(size):
                dx, dy = d.pop()
                mark[dx, dy] = markNum
                # 判断能不能扩展 4 方向
                for j in range(4):
                    nx, ny = dx + dir[j], dy + dir[j + 1]
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue
                    if mark[nx, ny] != 0:
                        continue
                    if img[nx, ny] > diffSho:
                        continue
                    d.append((nx, ny))

    # 参数：img 区分度图像
    # 返回 mark 分区域标记图像
    def regionMark(self, diffSho, img):
        h, w = img.shape
        mark = np.zeros(img.shape, np.uint8)
        markNum = 1
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if mark[i, j] == 0 and img[i, j] <= diffSho:
                    # 开始区域增长
                    self.BFS(mark, i, j, diffSho, img, markNum)
                    markNum += 1

        print("region grow ok\n")
        return mark

    def RegionGrowth(self, gray):
        diffSho, img = self.fullPicDiff(gray)
        mark = self.regionMark(diffSho, img)

        Tools().imgToCSV("regionMark", mark)
        binImg = mark.copy()
        h, w = mark.shape

        def check(i, j):
            if 0 <= i < h and 0 <= j < w:
                return True
            return False

        dir = [-1, 0, 1, 0, -1]
        for i in range(h):
            for j in range(w):
                flag = False
                for k in range(4):
                    dx, dy = i + dir[k], j + dir[k + 1]
                    if check(dx, dy) and mark[i, j] != mark[dx, dy]:
                        binImg[i, j] = 255
                        flag = True
                        break
                if flag == False:
                    binImg[i, j] = 0

        Tools().drawBinPic("regionExpand", binImg)

    # 不去除噪声的分区域方法
    def partitionABNew(self, gray, x, y, num=8):
        valueLists, indexLists = self.point_list(gray, x, y, num)
        print("\nLists={}".format(valueLists))
        A, B = [], []
        regionNum = 1
        ariseIndex, decIndex = -1, -1
        ariseMax, decMin = 0, 0
        for i in range(num):
            sub = int(valueLists[(i + 1) % num]) - int(valueLists[i])
            if sub > 0 and sub > ariseMax:
                ariseMax, ariseIndex = sub, i
            elif sub < 0 and sub < decMin:
                decMin, decIndex = sub, i
        minIndex, maxIndex = min(ariseIndex, decIndex), max(ariseIndex, decIndex)
        if minIndex == maxIndex:
            A = indexLists
            print("regionNum={}\tA={}\tB={}".format(regionNum, A, B))
            return regionNum, A, B
        # 否则， 有 2 个 区域
        regionNum = 2
        for i in range(num):
            if minIndex < i <= maxIndex:
                B.append(indexLists[i])
            else:
                A.append(indexLists[i])
        print("regionNum={}\tA={}\tB={}".format(regionNum, A, B))
        return regionNum, A, B

    # 无阈值 的区域标记
    def markWithNoSho(self, gray, num=8):
        h, w = gray.shape
        markNum = 1
        # secMaxMarkNum = 0
        dic = dict()
        markArr = np.zeros(gray.shape, int)

        def mark(indexRegion):
            nonlocal markNum, markArr, gray, dic
            if indexRegion == []:
                return
            s = set([markArr[i, j] for i, j in indexRegion])
            print(s)
            if len(s) == 1 and 0 in s:
                x, y = indexRegion[0]
                # if x == 73 and y == 42:
                #     input()
                if gray[x, y] in dic:
                    value = dic[gray[x, y]]
                    for i, j in indexRegion:
                        markArr[i, j] = value
                    return
                else:
                    for i, j in indexRegion:
                        markArr[i, j] = markNum
                    x, y = indexRegion[0]
                    dic[gray[x, y]] = markArr[x, y]
                    markNum += 1
                    # secMaxMarkNum += 1
            elif len(s) == 1 and 0 not in s:
                return
            else:

                if 0 in s:
                    if len(s) == 2:
                        for i, j in indexRegion:
                            if markArr[i, j] == 0:
                                x, y = i, j
                                break
                        if gray[x, y] in dic:
                            for i, j in indexRegion:
                                if markArr[i, j] == 0:
                                    markArr[i, j] = dic[gray[x, y]]
                            return

                        minV = 0x3fffffff
                        for v in s:
                            minV = min(minV, v) if v != 0 else minV

                        print(minV)

                        for i, j in indexRegion:
                            markArr[i, j] = minV if markArr[i, j] == 0 else markArr[i, j]
                    else:  # example {0, 2, 3}
                        # 有问题
                        l = list(s)
                        lens = len(s)
                        data = [0 for i in range(lens)]
                        for k, v in enumerate(l):
                            for i, j in indexRegion:
                                if v == markArr[i, j]:
                                    data[k] = gray[i, j]
                                    break
                        for i, j in indexRegion:
                            if markArr[i, j] == 0:
                                if abs(data[1] - gray[i, j]) < abs(data[2] - gray[i, j]):
                                    markArr[i, j] = l[1]
                                else:
                                    markArr[i, j] = l[2]
                        if data[1] == data[2]:
                            # if secMaxMarkNum + 1 == markNum:
                            # markNum -= 1
                            value = 0
                            if data[1] in dic:
                                value = dic[data[1]]
                            for i, j in indexRegion:
                                # markArr[i, j] = l[1] if markArr[i, j] != l[1] else l[1]

                                if markArr[i, j] != value:
                                    markArr[i, j] = value

                        # 方案 1 ： 把0 和 非 0 都换为最小
                        # markArr[i, j] = minV

                        # 方案2： 只将 0 替换为最小
                        # markArr[i, j] = minV if markArr[i, j] == 0 else markArr[i,j]

                        # 方案3: 0 最接近谁， 取谁
                        #       如果最大值和最小值相等， 最大值改为最小值
                # 没有 0
                else:
                    l = list(s)
                    for i, j in indexRegion:
                        if markArr[i, j] == l[0]:
                            data1 = gray[i, j]
                        if markArr[i, j] == l[1]:
                            data2 = gray[i, j]
                    if data1 == data2:
                        for i, j in indexRegion:
                            markArr[i, j] = l[0] if markArr[i, j] != l[0] else l[0]

        print('mark=\n{}'.format(markArr))
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                regionNum, A, B = self.partitionABNew(gray, i, j, num)
                mark(A)
                mark(B)
                # print('mark=\n{}'.format(markArr))
        self.drawEdgeWith8(markArr, num)
        return markArr

    def drawEdgeWith8(self, gray, num=8):
        # markArr = self.markWithNoSho(gray, num)
        h, w = gray.shape
        binImg = np.zeros(gray.shape, np.uint8)

        def check(i, j):
            if 0 <= i < h and 0 <= j < w:
                return True
            return False

        for i in range(h):
            for j in range(w):
                flag = False
                for r in range(-1, 2):
                    for c in range(-1, 2):
                        if r == 0 and c == 0:
                            continue
                        dx, dy = i + r, j + c
                        if check(dx, dy) and 0 != gray[dx, dy]:
                            binImg[i, j] = 255
                            flag = True
                            break
                    if flag:
                        break
                if flag == False:
                    binImg[i, j] = 0
        # Tools().imgToExcel(binImg, 'markArr', gray, markArr)
        # Tools().imgToExcel(binImg, 'markOrigin', gray)
        Tools().drawBinPic("regionExpand", binImg)

    def findTwoFalseNoise(self, gray, x, y, num=8):
        lists, _ = self.point_list(gray, x, y, num)
        noiseCost = []  # 8邻域每个点的价值

        # 单调 =》 true
        def check(left, mid, right):
            if left <= mid <= right or left >= mid >= right:
                return True
            else:
                return False

        for i in range(num):  # [0,7]
            lv, mv, rv = int(lists[i - 1]), int(lists[i]), int(lists[(i + 1) % num])
            if check(lv, mv, rv):
                cost = min(abs(lv - mv), abs(mv - rv))
                noiseCost.append(cost)
                # print("{}, {}, {} = a".format(lv, mv, rv))
            else:
                cost = abs(lv - mv) + abs(mv - rv) - abs(lv - rv)
                noiseCost.append(cost)
                # print("{}, {}, {} = b".format(lv, mv, rv))

        # 测试 每个点的能量值
        # print('\nnoiseEnergyLists={}'.format(noiseCost))
        # 在 noiseCost 数组 找最大值， 次大值
        max1, max2 = 0, 0
        index1, index2 = 0, 0

        for i in range(num):
            if noiseCost[i] > max2:
                if noiseCost[i] > max1:
                    max2, index2 = max1, index1
                    max1, index1 = noiseCost[i], i
                else:
                    max2, index2 = noiseCost[i], i
        # print("index1={}\tmax1={}\nindex2={}\tmax2={}".format(index1, max1, index2, max2))
        return index1, index2, max1, max2

    # paras：1.gray：原始灰度图 2. num 8邻域
    # return： 1. 噪声标记图 2.黑白图 3.噪声点个数
    def findTrueNoise(self, gray, num=8):
        h, w = gray.shape
        '''
        1. 取干扰能最大的 2 个点, 设为 A， B
        2. 求 3x3 矩阵的方差，设为E, 矩阵中心点设为 O
        3. if abs(AO) <= E and abs(BO) <= E, then no noise 
            elif abs(AO) > E and abs(BO) > E, then O is noise
            else
                if abs(AO) > E and abs(BO) <= E, then A is noise
                elif abs(BO) > E and abs(AO) <=E, then B is noise 
        '''

        noiseMark = np.zeros(gray.shape, np.uint8)

        def solveVariance(x0, y0, x1, y1):
            r = np.arange(x0, x1 + 1, 1)
            c = np.arange(y0, y1 + 1, 1)
            m = gray[r]
            m = m[:, c]
            # print('matrix=\n{}'.format(m))
            x = np.array(m)
            # E = np.var(x)
            E, V = 0, 0
            E = np.std(x, ddof=1)
            V = np.mean(x)
            # print('E={}'.format(E))
            return 0.2 * V + 0.9 * E

        noiseNum = 0
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                index1, index2, max1, max2 = self.findTwoFalseNoise(gray, i, j, num)
                # if max1 == 0 or max2 == 0:
                #     continue
                A, B = self.findIndex(i, j, index1, num), self.findIndex(i, j, index2, num)
                E = solveVariance(i - 1, j - 1, i + 1, j + 1)
                # print('A_index={}\tB_index={}'.format(index1, index2))
                # print('AVlaue={}\tBValue={}\tOvalue={}\n'.format(gray[A[0], A[1]], gray[B[0], B[1]], gray[i, j]))
                AO = abs(int(gray[A[0], A[1]]) - int(gray[i, j]))
                BO = abs(int(gray[B[0], B[1]]) - int(gray[i, j]))
                # print('AO={}\tBO={}'.format(AO, BO))
                # if max1 > 0 and max2 == 0 and AO == 0 and BO > 0:
                #     continue

                if AO <= E and BO <= E:
                    # print('no noise 2')
                    continue
                elif AO > E and BO > E:
                    noiseMark[i, j] += 1
                    noiseNum += 1
                    # print('O is noise 3')
                else:
                    if AO > E and BO <= E:
                        noiseMark[A[0], A[1]] += 1
                        noiseNum += 1
                        # print('A is noise 4')
                    elif BO > E and AO <= E:
                        noiseMark[B[0], B[1]] += 1
                        noiseNum += 1
                        # print('B is noise 5')

        # print('\nnoiseMarkMatrix=\n{}\n'.format(noiseMark))
        Tools().imgToExcel(2, 'noiseMark', noiseMark, gray)
        # print('\nours检测出的noiseNum={}\n'.format(noiseNum))
        binImg = np.zeros(gray.shape, np.uint8)
        for i in range(h):
            for j in range(w):
                if noiseMark[i, j] > 0:
                    binImg[i, j] = 255
                else:
                    binImg[i, j] = 0
        Tools().drawBinPic('edge', binImg)
        # self.drawEdgeWith8(noiseMark, num)
        return noiseMark, binImg

    # 统计图像边缘点的个数
    def calcEdgeNumber(self, gray, msg):
        edgeNum = 0
        h, w = gray.shape
        for i in range(h):
            for j in range(w):
                if gray[i, j] > 0:
                    edgeNum += 1
        print(msg + '图像的edge个数={}'.format(edgeNum))
        return edgeNum

    '''
    1. 生成 10 个椒盐噪声点， 并记录位置
    2. 不加噪声点跑出来的噪声标记设为 odd，加噪声点跑出来的噪声标记设为 new
    3. 对 new - odd，得到的 噪声标记找关系 
    '''

    # paras: 为 img 添加 num 个椒盐噪声
    # return: 1.添加噪声后的图像. 2. 添加噪声的坐标值
    def addJYNoise(self, img, num):
        h, w = img.shape
        img1 = img.copy()
        coordinateList = []
        for i in range(num):
            randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
            randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
            if (randx, randy) in coordinateList:
                continue
            coordinateList.append((randx, randy))
            if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
                img1[randx, randy] = 0
            else:
                img1[randx, randy] = 255
        return img1, coordinateList

    # 不同的位置记录下来
    #
    def newSubOdd(self, new, odd):
        h, w = new.shape
        # img = np.zeros(new.shape, np.uint8)
        indexList = []
        for i in range(h):
            for j in range(w):
                if new[i, j] != odd[i, j]:
                    # img[i,j] = 255
                    indexList.append((i, j))

        return indexList

    # 基于边连接的划分区域的方法
    def solves_11_13(self, gray, x, y, num=8):
        # 得到 （x，y）为中心的八邻域lists
        lists, _ = self.point_list(gray, x, y, num)
        print('\nmatrix={}'.format(lists))
        # 若严格单调，则价值为0 , 初始化 每个点 价值为0
        noiseCost = [0 for i in range(num)]  # 8邻域每个点的价值
        # A，B，C 是否为严格单调的标记数组，若是，将将B标记为 1 ，初始化，全部为 0
        monotoneList = [0 for i in range(num)]

        # 严格单调 =》 true
        def check(left, mid, right):
            if left <= mid <= right or left >= mid >= right:
                return True
            else:
                return False

        for i in range(num):  # [0,7]
            lv, mv, rv = int(lists[i - 1]), int(lists[i]), int(lists[(i + 1) % num])
            if check(lv, mv, rv):
                # monotoneList[i] = 1
                # cost = min(abs(lv-mv), abs(mv-rv))
                # noiseCost.append(cost)
                # print("{}, {}, {} = a".format(lv, mv, rv))
                cost = min(abs(lv - mv), abs(mv - rv))
                noiseCost[i] = cost
            else:
                cost = 2 * min(abs(lv - mv), abs(mv - rv))
                noiseCost[i] = cost
                # print("{}, {}, {} = b".format(lv, mv, rv))

        # 测试 每个点的能量值
        # print('\nnoiseEnergyLists={}'.format(noiseCost))
        # 在 noiseCost 数组 找最大值， 次大值
        max1, max2 = 0, 0
        index1, index2 = -1, -1

        for i in range(num):
            if noiseCost[i] > max2:
                if noiseCost[i] > max1:
                    max2, index2 = max1, index1
                    max1, index1 = noiseCost[i], i
                else:
                    max2, index2 = noiseCost[i], i
        # print("index1={}\tmax1={}\nindex2={}\tmax2={}".format(index1, max1, index2, max2))
        # return index1, index2, max1, max2
        # 去掉干扰能最大的点 和 严格单调的点
        # 去掉 index1 和 monotoneList[i] 列表为 1 的下标的值
        print('noiseCost={}'.format(noiseCost))
        print('monotoneList={}'.format(monotoneList))

        def checkMaxOrMin(bIndex):
            aIndex = (bIndex - 1 + num) % num;
            cIndex = (bIndex + 1) % num;
            if lists[aIndex] < lists[bIndex] > lists[cIndex] or lists[aIndex] > lists[bIndex] < lists[cIndex]:
                return True
            return False

        removeList = []
        for i in range(num):
            if (i == index1 and checkMaxOrMin(index1)) or monotoneList[i]:
                removeList.append(i)
        print('removeList={}'.format(removeList))
        valueResult = [lists[i] for i in range(num) if i not in removeList]
        indexResult = [i for i in range(num) if i not in removeList]
        print('valueResult={}\t indexResult={}'.format(valueResult, indexResult))
        return valueResult, indexResult

    def partitionAB_11_13(self, gray, x, y, num=8):
        valueResult, indexResult = self.solves_11_13(gray, x, y, num)
        # print("\nnoiseList={}\t noiseIndex={}".format(lists, index))
        A, B = [], []
        ariseIndex, decIndex = -1, -1
        ariseMax, decMin = 0, 0
        lens = len(valueResult)
        for i in range(lens):
            sub = int(valueResult[(i + 1) % lens]) - int(valueResult[i])
            # print("sub={}".format(sub))
            if sub > 0 and sub > ariseMax:
                ariseMax, ariseIndex = sub, i
            elif sub < 0 and sub < decMin:
                decMin, decIndex = sub, i
        # print("av={}\tai={}".format(ariseMax, ariseIndex))
        # print("dv={}\tdi={}".format(decMin, decIndex))
        minIndex, maxIndex = min(ariseIndex, decIndex), max(ariseIndex, decIndex)
        for i in range(lens):
            if minIndex < i <= maxIndex:
                B.append(indexResult[i])
            else:
                A.append(indexResult[i])
        print("A={}\tB={}".format(A, B))

        return A, B

    def fullPicture_11_13(self, gray, num=8):
        h, w = gray.shape
        # edgeMark = np.zeros(gray.shape, np.uint8)
        hArr = np.zeros(gray.shape, np.uint8)
        vArr = np.zeros(gray.shape, np.uint8)

        def check(x, y, A, B):
            hashA = np.zeros(num, np.uint8)
            for v in A:
                hashA[v] = 1
            hashB = np.zeros(num, np.uint8)
            for v in B:
                hashB[v] = 1

            '''
                对于A，B区域
                1) 如果未标记， 返回 -1
                2）如果有标记但不是全部标记， 返回标记 {0 - 7}
                3) 如果全部标记， 返回 8
            '''
            lenA, markA = -1, -1
            # if len(A) == 0:
            #     markA = -2
            # for i in A:
            #     xy = self.findIndex(x, y, i, num)
            #     markValue = edgeMark[xy]
            #     if markValue > 0:
            #         markA = i
            #         lenA += 1
            # if lenA == len(A):
            #     markA = 8
            #
            lenB, markB = -1, -1

            # if len(B) == 0:
            #     markB = -2
            # for i in B:
            #     xy = self.findIndex(x, y, i, num)
            #     markValue = edgeMark[xy]
            #     if markValue > 0:
            #         markB = i
            #         lenB += 1
            # if lenB == len(B):
            #     markB = 8

            def findNeighbor(i):
                return ((i - 1 + num) % num, (i + 1) % num)

            contactTupleSetA = set()
            contactTupleSetB = set()

            AB = (A, B)
            for i, lists in enumerate(AB):

                for v in lists:
                    oneTuple = findNeighbor(v)
                    for nei in oneTuple:
                        if i == 0:
                            if hashA[nei] == 0:
                                continue
                        else:
                            if hashB[nei] == 0:
                                continue
                        # contact v with nei
                        v1 = v
                        if v > nei:
                            v1, nei = nei, v
                        if i == 0:
                            contactTupleSetA.add((v1, nei))
                        else:
                            contactTupleSetB.add((v1, nei))
            print('markA={}\t'.format(markA))
            print('contactSetA={}\t'.format(contactTupleSetA))
            print('markB={}\t'.format(markB))
            print('contactSetB={}\t'.format(contactTupleSetB))
            return markA, contactTupleSetA, markB, contactTupleSetB

        markNum = 1
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                A, B = self.partitionAB_11_13(gray, i, j, num)
                markA, contactA, markB, contactB = check(i, j, A, B)
                '''
                h[i,j] = 1 represent h[i,j] and h[i, j+1] 有边相连
                v[i,j] = 1, 表示v[i, j] 和 v[i+1, j] 有边相连
                '''
                for (ii, jj) in contactA:
                    firCoor, secCoor = self.findIndex(i, j, ii, num), self.findIndex(i, j, jj, num)
                    if (0, 1) == (ii, jj):
                        hArr[firCoor] = 1
                    elif (1, 2) == (ii, jj):
                        hArr[firCoor] = 1
                    elif (4, 5) == (ii, jj):
                        hArr[secCoor] = 1
                    elif (5, 6) == (ii, jj):
                        hArr[secCoor] = 1
                    elif (2, 3) == (ii, jj):
                        vArr[firCoor] = 1
                    elif (3, 4) == (ii, jj):
                        vArr[firCoor] = 1
                    elif (6, 7) == (ii, jj):
                        vArr[secCoor] = 1
                    elif (0, 7) == (ii, jj):
                        vArr[firCoor] = 1
                for (ii, jj) in contactB:
                    firCoor, secCoor = self.findIndex(i, j, ii, num), self.findIndex(i, j, jj, num)
                    if (0, 1) == (ii, jj):
                        hArr[firCoor] = 1
                    elif (1, 2) == (ii, jj):
                        hArr[firCoor] = 1
                    elif (4, 5) == (ii, jj):
                        hArr[secCoor] = 1
                    elif (5, 6) == (ii, jj):
                        hArr[secCoor] = 1
                    elif (2, 3) == (ii, jj):
                        vArr[firCoor] = 1
                    elif (3, 4) == (ii, jj):
                        vArr[firCoor] = 1
                    elif (6, 7) == (ii, jj):
                        vArr[secCoor] = 1
                    elif (0, 7) == (ii, jj):
                        vArr[firCoor] = 1
                # if markA == -1:
                #     for (offsetA, offsetB) in contactA:
                #         xyA, xyB = self.findIndex(i, j, offsetA, num), self.findIndex(i, j, offsetB, num)
                #         edgeMark[xyA], edgeMark[xyB] = markNum, markNum
                #     markNum += 1
                # elif -1 < markA < 8:
                #     markValue = edgeMark[self.findIndex(i, j, markA)]
                #     for (offsetA, offsetB) in contactA:
                #         xyA, xyB = self.findIndex(i, j, offsetA, num), self.findIndex(i, j, offsetB, num)
                #         edgeMark[xyA], edgeMark[xyB] = markValue, markValue
                #
                # if markB == -1:
                #     for (offsetA, offsetB) in contactB:
                #         xyA, xyB = self.findIndex(i, j, offsetA, num), self.findIndex(i, j, offsetB, num)
                #         edgeMark[xyA], edgeMark[xyB] = markNum, markNum
                #     markNum += 1
                # elif -1 < markB < 8:
                #     markValue = edgeMark[self.findIndex(i, j, markB)]
                #     for (offsetA, offsetB) in contactB:
                #         xyA, xyB = self.findIndex(i, j, offsetA, num), self.findIndex(i, j, offsetB, num)
                #         edgeMark[xyA], edgeMark[xyB] = markValue, markValue

                # print('markArr={}'.format(edgeMark))
        binPic = np.zeros(gray.shape, np.uint8)

        # （i,j)与（i-1，j)有边相连，返回1， 否则返回 0
        def up(i, j):
            if i - 1 >= 0 and vArr[i - 1, j] == 0:
                return False
            return True

        def down(i, j):
            if i + 1 < h and vArr[i, j] == 0:
                return False
            return True

        def left(i, j):
            if j - 1 >= 0 and hArr[i, j - 1] == 0:
                return False
            return True

        def right(i, j):
            if j + 1 < w and hArr[i, j] == 1:
                return False
            return True

        def allHashEdge(i, j):
            if i - 1 >= 0 and vArr[i - 1, j] == 0:
                return False
            if i + 1 < h and vArr[i, j] == 0:
                return False
            if j - 1 >= 0 and hArr[i, j - 1] == 0:
                return False
            if j + 1 < w and hArr[i, j] == 0:
                return False
            return True

        def allNoEdge(i, j):
            if i - 1 >= 0 and vArr[i - 1, j] == 1:
                return False
            if i + 1 < h and vArr[i, j] == 1:
                return False
            if j - 1 >= 0 and hArr[i, j - 1] == 1:
                return False
            if j + 1 < w and hArr[i, j] == 1:
                return False
            return True

        for i in range(h):
            for j in range(w):
                # binPic[i,j] = 255
                if allHashEdge(i, j):
                    continue
                elif allNoEdge(i, j):
                    continue
                elif up(i, j) or down(i, j) or left(i, j) or right(i, j):
                    binPic[i, j] = 255

        # print(edgeMark)
        # binPic = Tools().markPicToBinPic4(edgeMark)
        Tools().imgToExcel(4, '11-13', binPic, gray, hArr, vArr)
        Tools().drawBinPic('11-13', binPic)
        # return edgeMark

    # 12.22
    def solve_12_22(self, gray, sho):
        self.solu(gray, sho);
        pass

    def solu(self, gray, sho):
        print(gray.dtype)
        gray = gray.astype(int)
        print(gray.dtype)

        h, w = gray.shape

        row = np.ones(gray.shape, np.uint8)
        # print(row)
        col = np.ones(gray.shape, np.uint8)

        for i in range(1, h - 2):
            for j in range(1, w - 2):
                # 计算横边的能量值
                Eab = abs(gray[i, j] - gray[i, j + 1])
                eList = []
                # 计算相邻6条边的能量值（未裁剪）
                # 1
                if col[i - 1, j] == 1:
                    E1 = abs(gray[i - 1, j] - gray[i, j])
                    eList.append(E1)
                # 2
                if col[i, j - 1] == 1:
                    E2 = abs(gray[i, j - 1] - gray[i, j])
                    eList.append(E2)
                # 3
                if col[i, j] == 1:
                    E3 = abs(gray[i + 1, j] - gray[i, j])
                    eList.append(E3)
                # 4
                if col[i - 1, j + 1] == 1:
                    E4 = abs(gray[i - 1, j + 1] - gray[i, j + 1])
                    eList.append(E4)
                # 5
                if col[i, j + 1] == 1:
                    E5 = abs(gray[i, j + 1] - gray[i, j + 2])
                    eList.append(E5)
                # 6
                if col[i, j + 1] == 1:
                    E6 = abs(gray[i, j + 1] - gray[i + 1, j + 1])
                    eList.append(E6)
                '''
                找最大边eMax与最小边eMin
                1. 如果Eab > eMax, 则断边 ##### 断边的情况1
                2. 否则， 如果Eab == eMax的前提下,
                            如果eMax == eMin（说明全部相等），不断边
                            如果eMax > eMin (说明Eab=eMax > eMin), 则断边 ###### 断边的情况2
                3. 否则， Eab < eMax, 不断边
                '''

                eMax, eMin = max(eList), min(eList)
                if Eab >= sho:
                    if Eab > eMax:
                        row[i, j] = 0
                    elif Eab == eMax and eMax > eMin:
                        row[i, j] = 0
                '''
                # 最开始的判断方法，发现行不通
                flag = True # 假设是局部最大边
                for v in eList:
                    if Eab < v:
                        flag = False
                        break
                if flag == True: # 为局部最大边
                    row[i, j] = 0 # 断边 
                '''

                # 计算竖边的能量值
                Eab = abs(gray[i, j] - gray[i + 1, j])
                eList = []
                # 计算相邻6条边的能量值（未裁剪）
                # 1
                if col[i - 1, j] == 1:
                    E1 = abs(gray[i - 1, j] - gray[i, j])
                    eList.append(E1)
                # 2
                if row[i, j - 1] == 1:
                    E2 = abs(gray[i, j - 1] - gray[i, j])
                    eList.append(E2)
                # 3
                if row[i, j] == 1:
                    E3 = abs(gray[i + 1, j] - gray[i, j])
                    eList.append(E3)
                # 4
                if row[i + 1, j - 1] == 1:
                    E4 = abs(gray[i + 1, j - 1] - gray[i + 1, j])
                    eList.append(E4)
                # 5
                if row[i + 1, j] == 1:
                    E5 = abs(gray[i + 1, j] - gray[i + 1, j + 1])
                    eList.append(E5)
                # 6
                if col[i + 1, j] == 1:
                    E6 = abs(gray[i + 1, j] - gray[i + 2, j])
                    eList.append(E6)

                eMax, eMin = max(eList), min(eList)
                if Eab >= sho:
                    if Eab > eMax:
                        col[i, j] = 0
                    elif Eab == eMax and eMax > eMin:
                        col[i, j] = 0

                '''
                flag = True  # 假设是局部最大边
                for v in eList:
                    if Eab < v:
                        flag = False
                        break
                if flag == True:  # 为局部最大边
                    col[i, j] = 0  # 断边
                '''
        print("row={}\n".format(row))
        print("col={}\n".format(col))
        # return row, col
        self.__pri_12_22(gray, row, col)
        pass

    def __pri_12_22(self, gray, row, col):
        h, w = row.shape
        binImage = np.zeros(row.shape, np.uint8)
        for i in range(1, h - 2):
            for j in range(1, w - 2):
                # 内部点 上下左右都右边相连
                if col[i - 1, j] == 1 and col[i, j] == 1 and row[i, j - 1] == 1 and row[i, j] == 1:
                    continue
                # 孤立点
                elif col[i - 1, j] == 0 and col[i, j] == 0 and row[i, j - 1] == 0 and row[i, j] == 0:
                    continue
                else:
                    binImage[i, j] = 1

        # Tools().drawBinPic('12_22', binImage)
        # Tools().imgToExcel(4, '11_22',binImage, gray, row, col)
        # return binImage # 1 表示边缘
        self.__secDrawEdge(binImage, gray, row, col)

    '''
    对于 binImg 二值图像，当边缘点周围还有边的时候，说明是真正的边缘
    '''

    def __secDrawEdge(self, binImg, gray, row, col):
        h, w = binImg.shape
        edgeImg = np.zeros(binImg.shape, np.uint8)
        '''
        (i,j)四周是否有边缘存在，存在返回True
        '''
        dir, dirLen = [-1, 0, 1, 0, -1], 4

        def __round_isEdge(i, j):
            for k in range(dirLen):
                dx, dy = i + dir[k], j + dir[k + 1]
                if binImg[dx][dy] == 1:
                    return True
            return False

        for i in range(1, h - 2):
            for j in range(1, w - 2):
                if binImg[i, j] == 1 and __round_isEdge(i, j):
                    edgeImg[i, j] = 1
        Tools().drawBinPic('12_22', edgeImg)
        Tools().imgToExcel(4, '11_22', edgeImg, gray, row, col)

    # 定义添加高斯函数
    def GaussianNoise(self, src, means, sigma, k):
        NoiseImg = src.copy()
        rows = NoiseImg.shape[0]
        cols = NoiseImg.shape[1]
        for i in range(rows):
            for j in range(cols):
                NoiseImg[i, j] = NoiseImg[i, j] + k * random.gauss(means, sigma)
                if NoiseImg[i, j] < 0:
                    NoiseImg[i, j] = 0
                elif NoiseImg[i, j] > 255:
                    NoiseImg[i, j] = 255
        return NoiseImg
