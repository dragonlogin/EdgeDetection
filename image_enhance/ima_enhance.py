from image_enhance.enhance_unit import enhance_unit
eu = enhance_unit()
import sys
import numpy as np
INF = sys.maxsize
class ima_enhance(object):
    def __init__(self):
        pass

    # test
    def get_AB(self, gray, x, y):
        return eu.partitionAB(gray, x, y)
    '''
    # 分为A，B两个区域， 中心点像素值离哪个区域的最小值近，就用那个区域的中值替代
    # 为主函数提供接口
    '''
    def replace_with_median(self, mat):
        h, w = mat.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = self.replace_with__(mat, i, j, 'median')
                print('i = {}\t, j = {}\t, val={}\n'.format(i, j, val))
                mat[i, j] = val

    # 获取data 列表中位数函数
    def get_median(self, data):
        return int(np.median(data))

    # 获取data 列表均值函数
    def get_mean(self, data):
        return int(np.mean(data))

    def replace_with__(self, mat, x, y, ttype):
        A, B = eu.partitionAB(mat, x, y)

        subA, subB = INF, INF
        # 如果某个区域为空，则令中心点与该区域的差值的绝对值为0

        if len(A) > 0:
            subA = abs(mat[x, y] - min(A))

        if len(B) > 0:
            subB = abs(mat[x, y] - min(B))

        if subA < subB:
            if ttype == 'median':
                return self.get_median(A)
            elif ttype == 'mean':
                return self.get_mean(A)
            else:
                return -1 # 表示错误

        else:
            if ttype == 'median':
                return self.get_median(B)
            elif ttype == 'mean':
                return self.get_mean(B)
            else:
                return -1 # 表示错误




