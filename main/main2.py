from functions.matrix.dis_degree import dis_degree
from functions.interference.interference import interference
from read_image.img_operator import img_operator
from read_image.image_path import  image_path
dd = dis_degree()
inter = interference()
io = img_operator()
igpath = image_path()

class main2(object):

    # 获取输入图像
    def get_input_mat(self):
        gray = io.readImg(igpath.getImagePath())
        #io.print('gray', gray)
        return gray
    # 获取区分度矩阵
    def get_dd_mat(self, ori_mat):
        return dd.from_originmat_to_dmat(ori_mat)

    # 获取干扰能矩阵
    def get_Intron_mat(self, ori_mat):
        return inter.get_interference_mat(ori_mat)


if __name__ == '__main__':
    obj = main2()
    ori_mat = obj.get_input_mat()
    # 一次区分度
    dd_mat = obj.get_dd_mat(ori_mat)
    io.showImg('dd_mat', dd_mat)

    # 二次区分度
    dd_mat2 = obj.get_dd_mat(dd_mat)
    io.showImg('dd_mat2', dd_mat2, True)


