from functions.matrix.dis_degree import dis_degree
from functions.interference.interference import interference
dd = dis_degree()
inter = interference()

class main2(object):

    # 获取区分度矩阵
    def get_dd_mat(self, ori_mat):
        return dd.from_originmat_to_dmat(ori_mat)

    # 获取干扰能矩阵
    def get_Intron_mat(self, ori_mat):
        return inter.get_interference_mat(ori_mat)

