from functions.matrix.check_mat import check_mat
cm = check_mat()
import numpy as np
class TL(object):
    def __init__(self):
        pass

    def t_10_21_mat2(self, mat2):
        # print('mat2=\n{}'.format(mat2))
        h, w = mat2.shape
        ret2 = mat2.copy()

        mat3 = cm.fenbian__(mat2)

        if cm.check4__(0, 0, mat3) == True:
            avg = np.mean(mat2)
            for i in range(h):
                for j in range(w):
                    ret2[i,j] = avg


        elif cm.check2__(0, 0, mat3) == True:
            edge3 = cm.mark_bsm2__(0, 0, mat3)
            x, y = 0, 0
            cl3 = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]
            cl2 = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
            # 0 3
            if edge3[cl3[0]] == 5 and edge3[cl3[3]] == 5:
                minval = min(mat2[cl2[0]], mat2[cl2[3]])
                ret2[cl2[0]] = ret2[cl2[3]] = minval
                maxval = max(mat2[cl2[1]], mat2[cl2[2]])
                ret2[cl2[1]] = ret2[cl2[2]] = maxval
                # 1 2
            elif edge3[cl3[1]] == 5 and edge3[cl3[2]] == 5:
                minval = min(mat2[cl2[1]], mat2[cl2[2]])
                ret2[cl2[1]] = ret2[cl2[2]] = minval
                maxval = max(mat2[cl2[0]], mat2[cl2[3]])
                ret2[cl2[0]] = ret2[cl2[3]] = maxval
                # 0 1
            elif edge3[cl3[0]] == 5 and edge3[cl3[1]] == 5:
                minval = min(mat2[cl2[0]], mat2[cl2[1]])
                ret2[cl2[0]] = ret2[cl2[1]] = minval
                maxval = max(mat2[cl2[3]], mat2[cl2[2]])
                ret2[cl2[3]] = ret2[cl2[2]] = maxval
                # 2 3
            elif edge3[cl3[2]] == 5 and edge3[cl3[3]] == 5:
                minval = min(mat2[cl2[2]], mat2[cl2[3]])
                ret2[cl2[2]] = ret2[cl2[3]] = minval
                maxval = max(mat2[cl2[0]], mat2[cl2[1]])
                ret2[cl2[0]] = ret2[cl2[1]] = maxval
            # print(edge3)


        elif cm.check3__(0, 0, mat3) == True:
            x, y = 0, 0
            cl3 = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]
            cl2 = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
            edge3 = cm.mark_bsm3__(0, 0, mat3)
            # 0
            if edge3[cl3[0]] == 6:
                minval = min(mat2[cl2[1]], mat2[cl2[2]], mat2[cl2[3]])
                ret2[cl2[1]] = ret2[cl2[2]] = ret2[cl2[3]] = minval
                # 1 2
            elif edge3[cl3[1]] == 6:
                minval = min(mat2[cl2[0]], mat2[cl2[2]], mat2[cl2[3]])
                ret2[cl2[0]] = ret2[cl2[2]] = ret2[cl2[3]] = minval

            elif edge3[cl3[2]] == 6:
                minval = min(mat2[cl2[0]], mat2[cl2[1]], mat2[cl2[3]])
                ret2[cl2[0]] = ret2[cl2[1]] = ret2[cl2[3]] = minval

            elif edge3[cl3[3]] == 6:
                minval = min(mat2[cl2[0]], mat2[cl2[1]], mat2[cl2[2]])
                ret2[cl2[0]] = ret2[cl2[1]] = ret2[cl2[2]] = minval
            # print(edge3)

        # print(ret2)
        return ret2

    def t_10_21_all_mat(self, in_mat):
        mat = in_mat.copy()
        h, w = mat.shape
        for i in range(h - 1):
            for j in range(w - 1):
                mat2 = mat[i : i + 2, j : j + 2]
                # print('mat2=\n{}'.format(mat2))
                ret2 = self.t_10_21_mat2(mat2)
                for ii in range(2):
                    for jj in range(2):
                        mat[i + ii][j + jj] = ret2[ii][jj]

        return mat

    def test_10_21(self):
        mat2 = np.array([
            [10,12],
            [24,22]
        ])
        mat2 = np.array([
            [22, 24],
            [10, 12]
        ])
        mat2 = np.array([
            [10, 24],
            [12, 22]
        ])
        mat2 = np.array([
            [22, 10],
            [24, 12]
        ])

        mat2 = np.array([
            [22, 10],
            [8, 12]
        ])
        mat2 = np.array([
            [10, 22],
            [8, 12]
        ])
        mat2 = np.array([
            [10, 12],
            [8, 22]
        ])
        mat2 = np.array([
            [10, 12],
            [22, 8]
        ])
        self.t_10_21(mat2)

# TL().test_10_21()

