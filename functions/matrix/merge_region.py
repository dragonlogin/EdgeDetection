from functions.matrix.check_mat import check_mat
import numpy as np
cm = check_mat()

'''
如果同一区域的最大区分度 大于 不同区域的的最大差，则代表是同一区域，合并
同一区域的最大区分度：同一区域max - 同一区域min
不同区域的最大差：大区域的最小值 - 小区域的最大值
'''
class merge_region(object):

    def merge3(self, mat_2) -> bool:
        x, y = 0, 0
        edge_3 = cm.fenbian__(x, y, mat_2)
        mat_3 = cm.mark_bsm3__(x, y, edge_3)
        # print('edge_3=\n{}'.format(edge_3))
        # print('\nmat_3=\n{}'.format(mat_3))

        pl2 = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        pl3 = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]
        be, se = [], []

        for i in range(4):
            if mat_3[pl3[i]] == 5:
                se.append(mat_2[pl2[i]])
            elif mat_3[pl3[i]] == 6:
                be.append(mat_2[pl2[i]])
        # print(be,se)

        if be == [] and se == []:
            return True

        max_sub, diff_val = 0, 0
        if len(be) == 3 and len(se) == 1:
            max_sub = max(be) - min(be)
            diff_val = min(be) - se[0]
        elif len(be) == 1 and len(se) == 3:
            max_sub = max(se) - min(se)
            diff_val = be[0] - max(se)

        if max_sub >= diff_val:
            return True

        return False

    def merge2(self, mat_2) -> bool:
        x, y = 0, 0
        edge_3 = cm.fenbian__(x, y, mat_2)
        mat_3 = cm.mark_bsm2__(x, y, edge_3)
        # print('edge_3=\n{}'.format(edge_3))
        # print('\nmat_3=\n{}'.format(mat_3))

        pl2 = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        pl3 = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]
        be, se = [], []

        for i in range(4):
            if mat_3[pl3[i]] == 5:
                se.append(mat_2[pl2[i]])
            elif mat_3[pl3[i]] == 6:
                be.append(mat_2[pl2[i]])
        # print(be,se)
        if be == [] and se == []:
            return True

        max_sub, diff_val = 0, 0
        if len(be) == 2 and len(se) == 2:
            max_sub = max(max(be) - min(be), max(se) - min(se))
            diff_val = min(be) - max(se)

        if max_sub >= diff_val:
            return True

        return False

    # test
    def merge3_test(self, mat_2):
        return self.merge3(mat_2)

    def main(self):
        gray = np.array([
            [50, 56, 56],
            [59, 52, 63],
            [187, 57, 44]
        ])
        for i in range(2):
            for j in range(2):
                b_val = self.merge3_test(gray[i:i + 2, j:j + 2])
                print(b_val)


