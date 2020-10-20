import numpy as np
class paper(object):

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
        return X[1 : h - 1][1 : w - 1]



   def test(self, mat):
       # 转到main/main2进行测试
       pass

















