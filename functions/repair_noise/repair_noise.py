from functions.interference.interference import interference

inter = interference()


class repair_noise(object):
    pass

    '''
        @:param 从(sx, sy)开始搜索局部极大值
        @:return  返回一个局部极大值坐标
        '''

    def find_local_max__(self, sx, sy, gray, d_mat, sho=0, bin_mat=[]):
        # 找局部极大值点
        h, w = gray.shape

        for i in range(sx, h):
            for j in range(sy, w):
                if d_mat[i, j] >= sho:
                    isT = True
                    for i_ in range(-1, 2):
                        if isT == False:
                            break
                        for j_ in range(-1, 2):
                            dx, dy = i_ + i, j_ + j
                            if (0 == i_ and 0 == j_) or dx < 0 or dx >= h or dy < 0 or dy >= w:
                                continue
                            if d_mat[i, j] < d_mat[dx, dy]:
                                isT = False
                                break
                    if isT == True:
                        # self.start_with(i, j, gray, d_mat, e_box)
                        return i, j

                if j + 1 == w:
                    sy = 0

        return -1, -1

    '''
    修复极大值点，极小值点，和干扰能极大值
    '''

    def repair_noise(self, mat):
        ret_mat = inter.cal_max_min_inter_max_6_14(mat)
        #
        # yellow = (128, 255, 255)
        # red = (0, 0, 255)
        # green = (0, 255, 0)
        # zise = (255, 0, 128)
        # qingse = (255, 255, 0)
        repair_mat = mat.copy()
        h, w = ret_mat.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if ret_mat[i, j] > 0:
                    mat3 = mat[i - 1: i + 2, j - 1: j + 2]
                    nx, ny = inter.get_nearest_coor(mat3)
                    repair_mat[i, j] = mat[nx + i - 1, ny + j - 1]

        return repair_mat
