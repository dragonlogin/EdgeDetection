import numpy as np
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from functions.matrix.check_mat import check_mat
cm = check_mat()


class reduce_md(object):

    #
    def pro_diff_edge__(self, flag, diff_val, edge_belong, mat_2, mat_3):
        pl2 = [(0, 0), (0, 1), (1, 1), (1, 0)]
        pl3 = [(0, 0), (0, 2), (2, 2), (2, 0)]

        if flag == 0:
            edge_belong.append(mat_3[pl3[2]])
        elif flag == 1:
            edge_belong.append(mat_3[pl3[3]])
        elif flag == 2:
            edge_belong.append(mat_3[pl3[0]])
        elif flag == 3:
            edge_belong.append(mat_3[pl3[1]])

        sl, bl = [], []

        for i in range(4):
            if mat_3[pl3[i]] == 5:
                sl.append(mat_2[pl2[i]])
            elif mat_3[pl3[i]] == 6:
                bl.append(mat_2[pl2[i]])

        s_max = max(sl)
        b_min = min(bl)
        diff_val.append(abs(int(s_max) - int(b_min)))

    def diff_edge_2x2__(self, flag, diff_val, edge_belong, mat_2):
        # print("mat_2=\n{}".format(mat_2))
        edge_mat_3 = cm.fenbian__(0, 0, mat_2)
        if cm.check4__(0, 0, edge_mat_3):
            return False

        else:
            if cm.check2__(0, 0, edge_mat_3):
                mark_mat_3 = cm.mark_bsm2__(0, 0, edge_mat_3)
                self.pro_diff_edge__(flag, diff_val, edge_belong, mat_2, mark_mat_3)
                return True
            elif cm.check3__(0, 0, edge_mat_3):
                mark_mat_3 = cm.mark_bsm3__(0, 0, edge_mat_3)
                self.pro_diff_edge__(flag, diff_val, edge_belong, mat_2, mark_mat_3)
                return True

    def add_sb_edgelist__(self, diff_val, edge_belong, s_el, b_el):
        if edge_belong[0] == 5:
            s_el.append(diff_val[0])
        elif edge_belong[0] == 6:
            b_el.append(diff_val[0])

    # cx, cy 是原始矩阵的中点坐标
    # gray 是原始mat 3x3
    # mark_arr 是标记mat
    # 如果(cx,cy)是矛盾点,调用此函数
    def cal_md_val(self, cx, cy, gray) -> int:
        h, w = gray.shape
        x, y = cx, cy
        # left up
        s_el, b_el = [], []

        if x - 1 >= 0 and y - 1 >= 0:
            diff_val, edge_belong = [], []
            mat_2 = gray[x - 1:x + 1, y - 1:y + 1]
            b_val = self.diff_edge_2x2__(0, diff_val, edge_belong, mat_2)
            if b_val:
                self.add_sb_edgelist__(diff_val, edge_belong, s_el, b_el)
        # right up
        if x - 1 >= 0 and y + 1 < w:
            diff_val, edge_belong = [], []
            mat_2 = gray[x - 1:x + 1, y:y + 2]
            b_val = self.diff_edge_2x2__(1, diff_val, edge_belong, mat_2)
            if b_val:
                self.add_sb_edgelist__(diff_val, edge_belong, s_el, b_el)
        # left down
        if x + 1 < h and y - 1 >= 0:
            diff_val, edge_belong = [], []
            mat_2 = gray[x:x + 2, y - 1:y + 1]
            b_val = self.diff_edge_2x2__(3, diff_val, edge_belong, mat_2)
            if b_val:
                self.add_sb_edgelist__(diff_val, edge_belong, s_el, b_el)
        # right down
        if x + 1 < h and y + 1 < w:
            diff_val, edge_belong = [], []
            mat_2 = gray[x:x + 2, y:y + 2]
            b_val = self.diff_edge_2x2__(2, diff_val, edge_belong, mat_2)
            if b_val:
                self.add_sb_edgelist__(diff_val, edge_belong, s_el, b_el)

        s_max, b_max = 257, 257

        if len(s_el) > 0:
            s_max = max(s_el)
        if len(b_el) > 0:
            b_max = max(b_el)

        r_val = min(s_max, b_max)

        return r_val

    # test
    def cal_md_val_test(self, cx, cy, gray):
        print('md_val=\t{}'.format(self.cal_md_val(cx, cy, gray)))

    # 传入一个numxnum的原始mat gray， 和mat的左上角坐标
    def cal_sho(self, gray, num=5) -> int:
        mark_arr = cm.bmsall_v2(gray)
        mh, mw = mark_arr.shape
        md_list = []

        for i in range(2, mh - 1, 2):
            for j in range(2, mw - 1, 2):
                if mark_arr[i, j] == 7:
                    cx, cy = i // 2, j // 2
                    gray_33 = gray[cx - 1:cx + 2, cy - 1:cy + 2]
                    # print('gray_3_3=\n{}'.format(gray_33))
                    md_val = self.cal_md_val(1, 1, gray_33)
                    md_list.append(md_val)
        sho = 0

        if len(md_list) > 0:
            sho = max(md_list)
        # print("md_list=\n{}".format(md_list))
        return sho

    def cal_sho_test(self, gray):
        self.cal_sho(gray)

    def main(self):
        gray = np.array([
            [50, 56, 56],
            [59, 52, 63],
            [187, 57, 44]
        ])
        # self.cal_md_val_test(1,1,gray)
        self.cal_sho_test(gray)


# test
# reduce_md().main()
