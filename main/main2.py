from functions.matrix.dis_degree import dis_degree
from functions.interference.interference import interference
from read_image.img_operator import img_operator
from read_image.image_path import  image_path
from image_enhance.ima_enhance import ima_enhance
from tools.tools import tools
from functions.add_noise.add_noise import add_noise
from image_enhance.paper import paper

from image_enhance.t_10_21 import TL
ta = TL()

from image_enhance.paper import paper_foma
foma = paper_foma()

an = add_noise()
import cv2
ie = ima_enhance()
dd = dis_degree()
inter = interference()
io = img_operator()
igpath = image_path()
tl = tools()
pp = paper()
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

    # 叉边方法
    def my_algo(self, mat, sho = 10):
        bin = dd.link_edge_all_mat_by_chabian(mat, sho)
        # bin = self.repair_bin(bin)
        tl.drawBinPicByChabian('my_algo', bin)

    def canny_algo(self, mat,sho1 = 50, sho2 = 120):
        can = cv2.Canny(mat, sho1, sho2)
        return can

    '''
    给图像添加噪声
    '''
    # 改进中值滤波，均值滤波测试
    def test_salt_noise(self, mat, num = 1):

        gray = an.addSaltNoise(mat, 0.9)
        # salt_after = gray - mat
        salt_after = gray
        tl.twoimgtoexcel('./sources/' + 'salt_after', mat, salt_after)
        cv2.imshow('origin_salt_noise', gray)
        '''
            改进中值滤波
        '''
        for i in range(num):
            r_mat = ie.replace_with_ttpye(gray, 'median')
            gray = r_mat
        io.showImg('salt_median', r_mat)
        tl.threeimgtoexcel('./sources/' + 'salt_blur_after', mat, salt_after, r_mat)

        cv2_img_salt_median = cv2.medianBlur(gray, 5)
        tl.threeimgtoexcel('./sources/' + 'cv2_img_salt_median', mat, salt_after, cv2_img_salt_median)
        io.showImg('cv2_salt_median', cv2_img_salt_median)
        '''
            改进均值滤波
        '''
        # for i in range(num):
        #     r_mat2 = ie.replace_with_ttpye(gray, 'mean')
        #     gray = r_mat2
        # io.showImg('salt_median', r_mat2)
        # tl.threeimgtoexcel('./sources/' + 'salt_mean_blur_after', mat, salt_after, r_mat2)


        # gray = an.GaussianNoise(mat, 0, 0.1, 8)
        # gaussian_after = gray
        # tl.twoimgtoexcel('./sources/' + 'gaussian_after', mat, gaussian_after)
        # # cv2.imshow('origin_gausssian_noise', gray)
        # #
        # for i in range(num):
        #     r_mat = ie.replace_with_ttpye(gray, 'median')
        #     gray = r_mat
        # tl.threeimgtoexcel('./sources/' + 'gaussian_blur_after', mat, gaussian_after, r_mat)
        # io.showImg('gaussian_mean', r_mat)
        #
        # for i in range(num):
        #     r_mat2 = ie.replace_with_ttpye(gray, 'mean')
        #     gray = r_mat2
        # io.showImg('gaussian_median', r_mat2)

    def test_salt_noise_wanfengfeng(self, mat, num=1):
        gray = an.addSaltNoise(mat, 0.03)
        # salt_after = gray - mat
        salt_after = gray
        tl.twoimgtoexcel('./sources/' + 'salt_after', mat, salt_after)
        cv2.imshow('origin_salt_noise', gray)

        wf_blur = pp.wanfeng(gray, 10, 30)
        cv2.imshow('wf_blur', wf_blur)

        '''
        2020-foma论文
        '''
        foma_blur = foma.foma(mat)
        cv2.imshow('foma_blur', foma_blur)

        '''
        SMF
        '''
        cv2_img_salt_median = cv2.medianBlur(gray, 5)
        # tl.threeimgtoexcel('./sources/' + 'cv2_img_salt_median', mat, salt_after, cv2_img_salt_median)
        io.showImg('cv2_salt_median', cv2_img_salt_median)

    def test_salt_noise_2020_foma(self, mat):
        gray = an.addSaltNoise(mat, 0.4)
        # salt_after = gray - mat
        salt_after = gray
        # tl.twoimgtoexcel('./sources/' + 'salt_after', mat, salt_after)
        cv2.imshow('origin_salt_noise', gray)

        '''
        2020-foma论文
        '''
        foma_blur = foma.foma(mat)
        cv2.imshow('foma_blur', foma_blur)

        '''
        SMF
        '''
        cv2_img_salt_median = cv2.medianBlur(gray, 5)
        # tl.threeimgtoexcel('./sources/' + 'cv2_img_salt_median', mat, salt_after, cv2_img_salt_median)
        io.showImg('cv2_salt_median', cv2_img_salt_median)

    def test_gaussian_noise(self, mat, num = 1):

        # gray = an.addSaltNoise(mat, 0.9)
        # salt_after = gray - mat
        # salt_after = gray
        # tl.twoimgtoexcel('./sources/' + 'salt_after', mat, salt_after)
        # cv2.imshow('origin_salt_noise', gray)
        # for i in range(num):
        #     r_mat = ie.replace_with_ttpye(gray, 'median')
        #     gray = r_mat
        # io.showImg('salt_mean', r_mat)

        # tl.threeimgtoexcel('./sources/' + 'salt_blur_after', mat, salt_after, r_mat)
        # for i in range(num):
        #     r_mat2 = ie.replace_with_ttpye(gray, 'mean')
        #     gray = r_mat2
        # io.showImg('salt_median', r_mat2)
        # tl.threeimgtoexcel('./sources/' + 'salt_mean_blur_after', mat, salt_after, r_mat2)


        gray = an.GaussianNoise(mat, 0, 0.1, 8)
        gaussian_after = gray
        tl.twoimgtoexcel('./sources/' + 'gaussian_after', mat, gaussian_after)
        # cv2.imshow('origin_gausssian_noise', gray)
        #
        for i in range(num):
            r_mat = ie.replace_with_ttpye(gray, 'median')
            gray = r_mat
        tl.threeimgtoexcel('./sources/' + 'gaussian_blur_after', mat, gaussian_after, r_mat)
        # io.showImg('gaussian_mean', r_mat)
        #
        # for i in range(num):
        #     r_mat2 = ie.replace_with_ttpye(gray, 'mean')
        #     gray = r_mat2
        # io.showImg('gaussian_median', r_mat2)

if __name__ == '__main__':
    obj = main2()
    ori_mat = obj.get_input_mat()
    io.showImg('ori_mat', ori_mat)
    # # 一次区分度
    # dd_mat = obj.get_dd_mat(ori_mat)
    # io.showImg('dd_mat', dd_mat)
    #
    # # 二次区分度
    # dd_mat2 = obj.get_dd_mat(dd_mat)
    # io.showImg('dd_mat2', dd_mat2, True)

    #对原图进行叉边方法
    # obj.my_algo(ori_mat, 15)
    # can = obj.canny_algo(ori_mat)
    # io.showImg('canny', can)



    # 对中值滤波
    # can = obj.canny_algo(r_mat)
    # io.showImg('canny_median', can)



    # 对均值滤波
    # can = obj.canny_algo(r_mat2)
    # io.showImg('canny_mean', can, True)

    # obj.test_salt_noise(ori_mat, 1)
    '''
        万丰丰论文实现
    '''
    obj.test_salt_noise_wanfengfeng(ori_mat)

    '''
    2020-foma 论文实现
    '''
    # obj.test_salt_noise_2020_foma(ori_mat)

    '''
        10-21讨论实现
    '''

    # ret_mat = ta.t_10_21_all_mat(ori_mat)
    # io.showImg('t_10_21', ret_mat)
    # tl.twoimgtoexcel('./sources/' + 't_10_21', ori_mat, ret_mat)

    cv2.waitKey(0)

