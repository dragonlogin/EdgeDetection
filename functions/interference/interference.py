import numpy as np
from functions.matrix.dis_degree import dis_degree

dd = dis_degree()


class interference(object):

    def get_inter_mat2__(self, cx, cy, mat):
        O = (cx, cy)
        A, B, C, D = (cx, cy - 1), (cx - 1, cy), (cx, cy + 1), (cx + 1, cy)
        enum_list = [
            [A, O, B],
            [B, O, C],
            [C, O, D],
            [D, O, A],
            [A, O, C],
            [B, O, D]
        ]
        E = 0
        for (ap, hp, bp) in enum_list:
            a, h, b = int(mat[ap]), int(mat[hp]), int(mat[bp])

            tmp = 0
            if (h >= a and h >= b) or (h <= a and h <= b):
                tmp = 2 * min(abs(h - a), abs(h - b))
            else:
                tmp = abs(b - a) + min(abs(h - a), abs(h - b))
            # print(a, h, b, tmp)
            E = max(E, tmp)

        # print('E', E)
        return E

    def get_inter_mat2__test(self):
        mat = np.array([
            [1, 30, 1],
            [20, 26, 34],
            [1, 41, 1]
        ])
        # self.get_inter_mat2__(1, 1, mat)
        self.get_interference_mat()

    def cal_max_min_inter_max_6_14(self, mat):
        local_min_mat = dd.process_local_mat(mat, 1, False)
        local_max_mat = dd.process_local_mat(mat, 10, True)
        inter_mat = self.get_interference_mat(mat)
        inter_local_max_mat = dd.process_local_mat(inter_mat, 100, True)

        ret_mat = np.zeros(mat.shape, dtype=np.uint8)

        ret_mat += local_max_mat
        ret_mat += local_min_mat
        ret_mat += inter_local_max_mat

        # tl.drawBinPic('local_min', local_min_mat)
        # tl.drawBinPic('local_max', local_max_mat)
        # tl.drawBinPic('inter_max', inter_local_max_mat)
        # self.show_max_min_inter_max_rgb(ret_mat, 'black', False)
        # tl.min_max_inter_max_to_excel('min_max_inter_mat', mat, ret_mat)
        return ret_mat

    def get_interference_mat(self, mat):
        h, w = mat.shape
        inter_mat = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                inter_val = self.get_inter_mat2__(i, j, mat)
                inter_mat[i, j] = inter_val

        return inter_mat

    def get_nearest_coor(self, mat3):

        ox, oy = 0, 0  # offset
        c_val = (int)(mat3[ox + 1, oy + 1])
        coor_list = [(ox, oy), (ox, oy + 1), (ox, oy + 2),
                     (ox + 1, oy + 2), (ox + 2, oy + 2),
                     (ox + 2, oy + 1), (ox + 2, oy),
                     (ox + 1, oy)
                     ]
        nx, ny = 0, 0
        dist = 256
        for i in coor_list:
            pix = (int)(mat3[i])
            print(i, pix)
            tmp = abs(c_val - pix)
            if tmp < dist:
                dist = tmp
                nx, ny = i

        print('nearest coor', nx, ny)
        return (nx, ny)

    def main(self):
        # self.get_inter_mat2__test()
        mat = np.array([
            [1, 30, 1],
            [20, 26, 34],
            [1, 41, 1]
        ])

        # mat = np.array([
        #     [1, 213, 1],
        #     [212, 214, 211],
        #     [1, 215, 1]
        # ])
        # self.get_interference_mat(mat)
        self.get_nearest_coor(mat)

# it = interference()
# it.main()
