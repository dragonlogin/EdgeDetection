import numpy as np
import math
class paper_wff(object):

   def __init__(self):
        pass

   '''
        万丰丰paper（AFMF）
   '''
   def wanfeng(self, mat, T1, T2):
        h, w = mat.shape
        h, w = h + 2, w + 2
        X = np.zeros((h, w), dtype=np.uint8)

        X[0, 0] = mat[0, 0]
        X[h - 1, 0] = mat[h - 3, 0]
        X[0, w - 1] = mat[0, w - 3]
        X[h - 1, w - 1] = mat[h - 3, w - 3]

        # 缝补第一行和最后一行
        for i in range(1, w - 1):
            X[0, i] = mat[0, i - 1]
            X[h - 1, i] = mat[h - 3][i - 1]

        # 缝补第一列和最后一列
        for i in range(1, h - 1):
            X[i, 0] = mat[i - 1, 0]
            X[i, w - 1] = mat[i - 1, w - 3]

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                X[i, j] = mat[i - 1, j - 1]

        '''
            极值法
            :param val 坐标(i,j)的像素值
            :return true ： 疑似噪声点
            false : 非噪声点
        '''
        def extreme(val):
            return True if val == 0 or val == 255 else False

        '''
        :param (i, j)坐标
        :return (i,j)到集合S的元素差的绝对值的平均值，其中S={(i-1,j-1), (i-1,j),(i-1,j+1), (i, j - 1)}坐标像素值
        '''
        def D_5(i, j):
            ls = []
            cor = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1)]
            for (cx, cy) in cor:
                ls.append(abs(int(X[i, j]) - int(X[cx, cy])))

            return np.mean(ls)

        '''
            公式6
        '''
        def F_6(d5_ret, t1, t2):
            ret = 0.0
            if d5_ret < t1:
                ret = 0.0
            elif d5_ret >= t2:
                ret = 1.0
            else:
                ret = float((d5_ret - t1) * 1.0 / (t2 - t1))

            return float(ret)

        '''
            统计当前滤波窗口信号点的数量 和信号点的平均值
        '''
        def blur_win(ci, cj, r):
            cnt = 0
            sum = 0
            for i in range(ci - r // 2, ci + r // 2 + 1):
                for j in range(cj - r // 2, cj + r // 2 + 1):
                    if i < 0 or i >= h or j < 0 or j >= w:
                        continue
                    if extreme(X[i, j]) == False:
                        cnt += 1
                        sum += X[i, j]

            if cnt > 0:
                sum /= cnt
            return cnt, sum

        '''
            滤波估计值函数y( i, j )=( 1 - F( i, j ) ) * x( i, j )+ F( i, j ) * M( i, j )
        '''
        def Y_9(ci, cj, f_val, m_val):
            ret = int((1 - f_val) * X[ci, cj] + f_val * m_val)
            return ret

        '''
            公式10
        '''
        def F_10(i, j):
            return D_5(i, j)


        '''
            整个算法流程
        '''

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # (1)第一步
                if extreme(X[i, j]) == False:
                    continue
                d5_ret = D_5(i, j)
                f6_ret = F_6(d5_ret, T1, T2)

                eps = 0.000001
                if f6_ret <= eps:
                    continue

                # 步骤4 滤波阶段
                r = 3
                while r <= 7:
                    cnt, avg = blur_win(i, j, r)
                    if cnt == 0:
                        r += 2
                        continue

                    tmp = Y_9(i, j, f6_ret, avg)
                    X[i, j] = tmp
                    break

                if r > 7:
                    X[i, j] = F_10(i, j)

        # a[[1, 4]][:, [0, 3]]
        return X[1 : h - 1, 1 : w - 1]



   def test(self, mat):
       # 转到main/main2进行测试
       pass



class paper_foma(object):
    def __init__(self):
        pass

    '''
    输入：mat, mat如果是3x3就是对3x3求，如果是5x5就是对5x5求
    输出：一个值
    '''
    def TMF(self, mat):
        h, w = mat.shape
        num, ls = 0, []
        for i in range(h):
            for j in range(w):
                pix = mat[i, j]
                if 0 < pix < 255:
                    num += 1
                    ls.append(pix)

        if num == 0:
            return mat[h // 2, w // 2]
        else:
            return math.ceil(np.median(ls))

    '''
    输入：3x3 mat
    输出：一个值
    '''
    def ARA(self, mat3):
        i, j = 1, 1
        cor = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1)]
        sum = 0

        for k in cor:
            sum += mat3[k]

        return math.ceil(sum // 4)

    '''
    预处理
    输入：mat
    输出：边缘填充两层的0
    '''
    def init(self, mat):
        h, w = mat.shape
        ret = np.zeros((h + 4, w + 4), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                ret[i + 2, j + 2] = mat[i, j]

        return ret

    '''
    判断一个点是否为疑似噪声点
    输入：一个值
    输出：True/False
    '''
    def is_noisy(self, val):
        return False if 0 < val < 255 else True

    '''
    算法第一阶段
    输入：一个含有椒盐噪声的 mat
    输出：第一阶段后的图像 img1
    '''
    def step_one(self, dmat):
        rimg1 = dmat.copy()
        h, w = dmat.shape
        h, w = h - 4, w - 4

        # 阶段一
        for i in range(h):
            for j in range(w):
                ni, nj = i + 2, j + 2

                if self.is_noisy(dmat[ni, nj]) == False:
                    continue

                mat3 = dmat[ni - 1: ni + 2, nj - 1: nj + 2]
                mat5 = dmat[ni - 2: ni + 3, nj - 2: nj + 3]
                rtmf3 = self.TMF(mat3)
                rtmf5 = self.TMF(mat5)

                if 0 < rtmf3 < 255:
                    rimg1[ni, nj] = rtmf3
                else:
                    rimg1[ni, nj] = rtmf5

        return rimg1

    '''
    算法第二阶段
    输入：第一阶段的 imag1
    输出：第二阶段后的图像的 img2 
    '''

    def step_two(self, rimg1):
        rimg2 = rimg1.copy()
        h, w = rimg1.shape
        h, w = h - 4, w - 4

        for i in range(h):
            for j in range(w):
                ni, nj = i + 2, j + 2

                if self.is_noisy(rimg1[ni, nj]) == False:
                    continue

                mat5 = rimg1[ni - 2: ni + 3, nj - 2: nj + 3]
                rtmf5 = self.TMF(mat5)
                rimg2[ni, nj] = rtmf5

        return rimg2

    '''
    算法的第三阶段
    输入：第二阶段的结果 rimg2
    输出：第三阶段的结果 rimg3
    '''
    def step_three(self, rimg2):
        rimg3 = rimg2.copy()
        h, w = rimg2.shape
        h, w = h - 4, w - 4

        for i in range(h):
            for j in range(w):
                ni, nj = i + 2, j + 2

                if self.is_noisy(rimg2[ni, nj]) == False:
                    continue

                mat3 = rimg2[ni - 1: ni + 2, nj - 1: nj + 2]
                ara = self.ARA(mat3)
                rimg3[ni, nj] = ara

        return rimg3

    '''
    算法第四阶段：处理边界像素
    输入：第三阶段的结果 rimg3
    输出：第四阶段的结果 rimg4
    '''
    def step_four(self, rimg3):
        h, w = rimg3.shape
        return rimg3[2 : h - 2, 2 : w - 2]


    '''
    论文算法流程
    输入：一个含有椒盐噪声的mat
    输出：一个滤波后的mat
    '''


    def foma(self, mat):
        # 待处理的 dmat
        dmat = self.init(mat)

        #第一阶段
        rimg1 = self.step_one(dmat)
        # 阶段二
        rimg2 = self.step_two(rimg1)
        # 阶段三
        rimg3 = self.step_three(rimg2)

        # 阶段四
        rimg4 = self.step_four(rimg3)
        return rimg4



    def test(self):
        ls = [1, 2, 3, 4, 5, 6, 7]
        import math
        m = math.ceil(np.median(ls))
        print(m)

    def test_foma(self):
        mat = np.array([
            [0, 0, 255, 255, 255, 0, 255],
            [125, 127, 0, 255, 0, 255, 255],
            [126, 127, 255, 0, 255, 0, 255],
            [126, 129, 0, 255, 255, 0, 255],
            [127, 129, 0, 255, 255, 0, 255]
        ])

        print('mat=\n{}\n'.format(mat))
        #预处理
        dmat = self.init(mat)
        print('dmat=\n{}\n'.format(dmat))

        # 第一阶段
        rimg1 = self.step_one(dmat)
        print('rimg1=\n{}\n'.format(rimg1))

        # 第二阶段
        rimg2 = self.step_two(rimg1)
        print('rimg2=\n{}\n'.format(rimg2))

        # 第三阶段
        rimg3 = self.step_three(rimg2)
        print('rimg3=\n{}\n'.format(rimg3))

        # 第四阶段
        rimg4 = self.step_four(rimg3)
        print('rimg4=\n{}\n'.format(rimg4))

        rimg4_foma = self.foma(mat)
        print('rimg4_foma=\n{}\n'.format(rimg4_foma))


# paper_foma().test_foma()
class MDBUTM_2011(object):
    def init(self):
        pass


    def is_noisy(self, val):
        return True if 0 < val < 255 else False

    '''
    此方法前提是mat3中心是噪声，然后用返回值替换
    '''
    def replace_val(self, mat3):
        h, w = mat3.shape

        cnt, ls = 0, []
        for i in range(h):
            for j in range(w):
                val = mat3[i, j]

                if self.is_noisy(val) == True:
                    cnt += 1
                    ls.append(val)

        if cnt == 0:
            return math.ceil(np.mean(mat3))
        else:
            return math.ceil(np.median(np.array(ls)))


    '''
    MDBUTM论文方法处理整副图像
    '''
    def mdbutm(self, mat):
        h, w = mat.shape
        ret_mat = mat.copy()

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = mat[i, j]

                if self.is_noisy(val) == True:
                    continue

                mat3 = mat[i - 1 : i + 2, j - 1 : j + 2]
                ret_mat[i, j] = self.replace_val(mat3)

        return ret_mat



    def test_mdbutm(self):
        mat = np.array([
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 255]
        ])
        mat = np.array([
            [78, 90, 0],
            [120, 0, 255],
            [97, 255, 73]
        ])
        mat = np.array([
            [43, 67, 70],
            [55, 90, 79],
            [85, 81, 66]
        ])

        print('mat = \n{}\n'.format(mat))
        self.mdbutm(mat)
        print('mdbutm_mat = \n{}\n'.format(mat))



# MDBUTM_2011().test_mdbutm()









