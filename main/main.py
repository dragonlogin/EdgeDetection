from functions.matrix.dis_degree import dis_degree
from read_image.img_operator import img_operator
from tools.tools import tools
from read_image.image_path import  image_path
from functions.matrix.check_mat import check_mat
from functions.interference.interference import interference
from functions.add_noise.add_noise import add_noise
from functions.repair_noise.repair_noise import repair_noise
# from functions.Test.Test import Test
from functions.repair_md.repair_md import repair_md
from functions.measure_method.measure import measure
ms = measure()
rmd = repair_md()
# t = Test()
rn = repair_noise()
an = add_noise()
inter = interference()
cm = check_mat()
tl = tools()
dd = dis_degree()
io = img_operator()
igpath = image_path()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class DDTest(object):

    def __init__(self):
        self.gray = np.array([
            [144, 140, 135, 137, 138, 140, 139],
            [113, 150, 142, 139, 141, 138, 138],
            [32, 88, 145, 152, 139, 137, 138],
            [22, 21, 63, 132, 156, 139, 136],
            [30, 27, 18, 44, 110, 150, 148],
            [27, 30, 24, 22, 23, 63, 136],
            [17, 21, 27, 27, 25, 28, 42],
            [22, 26, 28, 29, 31, 27, 27],
            [25, 29, 26, 27, 30, 27, 26],
            [24, 25, 25, 25, 26, 27, 29],
            [23, 21, 24, 27, 26, 25, 24],
            [24, 23, 23, 24, 24, 26, 23]
        ])

        self.mat = np.array([
            [126, 123, 125],
            [125, 122, 124],
            [126, 123, 124]
        ])

        self.mat = np.array([
            [134, 136, 122],
            [65, 98, 124],
            [21, 23, 40]
        ])

        self.mat = np.array([
            [35, 33, 30, 17, 9, 43],
            [25, 25, 27, 21, 50, 125],
            [22, 15, 27, 54, 105, 118],
            [102, 98, 108, 125, 118, 112],
            [103, 113, 116, 109, 114, 108],
            [110, 113, 111, 111, 108, 115]
        ])
        self.mat = np.array([
            [22, 30, 27, 29, 23],
            [78, 23, 26, 30, 28],
            [140, 82, 26, 24, 30],
            [137, 143, 88, 27, 23],
            [125, 137, 140, 92, 33]
        ])

        self.mat = np.array([
            [130, 130, 131, 132, 133, 133, 133, 132],
            [130, 132, 131, 131, 132, 132, 133, 133],
            [131, 131, 130, 130, 130, 131, 132, 132],
            [130, 130, 130, 129, 129, 131, 131, 131],
            [128, 129, 129, 128, 129, 131, 131, 130],
            [126, 127, 127, 127, 129, 130, 130, 128],
            [127, 127, 125, 127, 129, 127, 128, 129],
            [126, 127, 124, 126, 128, 126, 127, 128],
            [125, 126, 123, 125, 127, 126, 127, 126],
            [124, 125, 122, 124, 127, 126, 126, 126],
            [124, 126, 123, 124, 126, 125, 125, 124],
            [124, 126, 123, 123, 125, 123, 123, 122],
            [122, 125, 123, 123, 123, 122, 123, 122],
            [120, 124, 122, 122, 123, 122, 123, 122],
            [122, 122, 122, 122, 122, 122, 122, 122]
            ]
        )


    # truthPath = '../../image/8068.mat'
    def get_gray(self):

        gray = io.readImg(igpath.getImagePath())
        io.print('gray', gray)
        return gray

    def get_groundtruth_mat(self):
        ground_mat = io.returnGroundTruthMat(igpath.getMatPath())
        cv2.imshow('groundTruth', ground_mat)
        return ground_mat


    def remove_extra_all_mat_test(self):
        gray = self.get_gray()
        ground_mat = self.get_groundtruth_mat()
        # gray = self.gray
        # d_mat, mark, dv_mat = dd.remove_extra_all_mat(gray, 3)
        e_mat_before = dd.distinction_degree_elimination_allmat_before(gray)
        d_mat = dd.from_originmat_to_dmat(gray)
        dd_mat = dd.combine_e_d(e_mat_before, d_mat)
        e_mat_after = dd.distinction_degree_elimination_allmat_after(gray, dd_mat)
        sho_mat = dd.get_sho_mat(gray)
        # io.print('mark', mark)
        # tl.gray_dmat_mark_sho_to_excel('gray_dif_sho', gray, d_mat, e_mat_after, sho_mat, ground_mat)
        tl.gray_dmat_sho_to_excel('gray_dif_sho_ebefore_eafter', gray, d_mat, e_mat_before, e_mat_after, sho_mat, ground_mat)
        tl.drawBinPic('e_mat_before', e_mat_before)
        tl.drawBinPic('e_mat_after', e_mat_after)

    def new_think(self, sho):
        gray = self.get_gray()
        d_mat = dd.from_originmat_to_dmat(gray)
        sho_mat = dd.get_sho_mat(gray)
        mark_mat = np.zeros(gray.shape, dtype=np.uint8)
        ground_mat = self.get_groundtruth_mat()
        h, w = gray.shape
        for i in range(h):
            for j in range(w):
                if d_mat[i, j] > sho and d_mat[i, j] > sho_mat[i, j]:
                    mark_mat[i, j] = 255
        tl.gray_dmat_mark_sho_to_excel('gray_dif_sho_new_think', gray, d_mat, mark_mat, sho_mat, ground_mat)
        cv2.imshow('new_think', mark_mat)

    # def splice_gray(self, mat, x0, y0, x1, y1):
    #     return mat[]

    def repair_bin(self, bin):
        h, w = bin.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if bin[i, j] == 0:
                    up_v, down_v, left_v, right_v = bin[i-1, j], bin[i+1,j], bin[i, j-1], bin[i, j+1]
                    if (up_v>0 and up_v == down_v==1) or (left_v>0 and left_v == right_v==1):
                        bin[i,j] = 1
                    elif (up_v>0 and up_v == down_v==10) or (left_v>0 and left_v == right_v==10):
                        bin[i,j] = 10

        return bin

    def single_pic_show_rgb(self, bin_mat):
        three_mat = cv2.imread('../' + igpath.getImagePath())
        tmp = three_mat.copy()

        h, w = bin_mat.shape

        yellow = (128, 255,255)
        for i in range(h):
            for j in range(w):
                tmp[i, j] = (0, 0, bin_mat[i, j])

        cv2.imshow('single_bin', tmp)

    def show_canny_and_origin_rgb(self, can, background_color, need_background = True):
        gray = cv2.imread(igpath.getImagePath())
        # three_mat = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
        # tmp = three_mat.copy()
        tmp = gray.copy()
        h, w = can.shape

        yellow = (128, 255,255)
        red = (0, 0 , 255)
        green = (0, 255, 0)
        juhong = (255, 0, 255)

        black_background = (0, 0, 0)
        white_background = (255,255,255)
        for i in range(h):
            for j in range(w):
                if can[i, j] > 0:
                    tmp[i, j] = juhong
                else:
                    if need_background == False:
                        if background_color == 'black':
                            tmp[i, j] = black_background
                        elif background_color == 'white':
                            tmp[i, j] = white_background




        cv2.imshow('canny_edge', tmp)

    def show_max_min_inter_max_rgb(self, bin, background_color, need_background = True):
        three_mat = cv2.imread('../' + igpath.getImagePath())
        tmp = three_mat.copy()

        h, w = bin.shape

        yellow = (128, 255,255)
        red = (0, 0 , 255)
        green = (0, 255, 0)
        zise = (255, 0, 128)
        qingse = (255, 255, 0)

        black_background = (0, 0, 0)
        white_background = (255,255,255)

        for i in range(h):
            for j in range(w):
                if bin[i, j]  == 1:
                    tmp[i, j] = yellow
                elif bin[i, j] == 10:
                    tmp[i, j] = red
                elif bin[i, j] == 100:
                    tmp[i, j] = green
                elif bin[i, j] == 101:
                    tmp[i, j] = qingse
                elif bin[i, j] == 110:
                    tmp[i, j] = zise
                else:
                    if need_background == False:
                        if background_color == 'black':
                            tmp[i, j] = black_background
                        elif background_color == 'white':
                            tmp[i, j] = white_background




        cv2.imshow('local_max_min_inter_max', tmp)

    def show_myalgo_and_origin_pic_rgb(self, bin, background_color, need_background = True):
        three_mat = cv2.imread( igpath.getImagePath())
        # three_mat = cv2.resize(three_mat, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
        tmp = three_mat.copy()

        h, w = bin.shape

        yellow = (128, 255,255)
        red = (0, 0 , 255)
        green = (0, 255, 0)
        juhong = (255, 0, 255)

        black_background = (0, 0, 0)
        white_background = (255, 255, 255)

        for i in range(h):
            for j in range(w):
                if 0 < bin[i, j] < 10:  # 红色小边
                    tmp[i, j] = red
                elif bin[i, j] >= 10:
                    tmp[i, j] = green
                else:
                    if need_background == False:
                        if background_color == 'black':
                            tmp[i, j] = black_background
                        elif background_color == 'white':
                            tmp[i, j] = white_background

        cv2.imshow('myalgo_edge', tmp)


    def show_rgb(self, bin, can, inter_mat = []):
        three_mat = cv2.imread( igpath.getImagePath())
        # print(three_mat)
        tmp = three_mat.copy()

        h, w = bin.shape

        yellow = (128, 255,255)
        red = (0, 0 , 255)
        green = (0, 255, 0)
        juhong = (255, 0, 255)

        for i in range(h):
            for j in range(w):
                if inter_mat[i, j] > 20:
                    tmp[i, j] = yellow

                if can[i, j] > 0:
                    tmp[i, j] = juhong

                if 0 < bin[i, j] < 10:  # 红色小边
                    tmp[i, j] = red
                elif bin[i, j] >= 10:
                    tmp[i, j] = green
                    # if inter_mat[i, j] > 20:
                    #     tmp[i, j] = (100, 0, inter_mat[i, j])
                    # else:
                    #     tmp[i, j] = (inter_mat[i,j], 0, 0)
                    # tmp[i, j] = (0, 0, 0)
                # if ground_mat[i, j] > 0:
                #     tmp[i, j] = (255, 0, 0)


        # new_im =
        # cv2.imshow('leaf', three_mat)
        cv2.imshow('three_edge', tmp)

    '''
        像素局部极小值 标记为1
        像素局部极大值 标记为10
        像素局部极大值 标记为100
        '''


    def adjust_0_255(self, binImg):
        h, w = binImg.shape
        img = binImg.copy()
        for i in range(h):
            for j in range(w):
                if 0 < binImg[i, j] < 10:
                    img[i, j] = 255
                else:
                    img[i, j] = 0

        return img
    def show_diff(self, sho = 15):
        mat = self.get_gray()
        t.diff_of_pre_repair_and_sur_repair(mat, sho)

    def new_think2(self, sho):
        gray = self.get_gray()

        # gray = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('./DestImage/girl.jpg', gray)
        # cv2.imshow('originGray', gray)

        # blur_res = cv2.GaussianBlur(gray, (5,5), 0)
        #
        # x0, x1, y0, y1 = 194, 213, 290, 306
        #
        # gray = gray[x0: x1, y0: y1]
        # gray = gray[55 : 59, 9 : 13]
        # print('gray', gray)
        '''
            获取干扰能矩阵 
        '''
        inter_mat = inter.get_interference_mat(gray)
        cv2.imshow('inter_mat', inter_mat)

        '''
        显示像素值局部极大，极小和干扰能局部极大 
        '''
        ret_mat = inter.cal_max_min_inter_max_6_14(gray)
        # tl.min_max_inter_max_to_excel('min_max_inter_mat', gray, ret_mat, inter_mat)
        # self.show_max_min_inter_max_rgb(ret_mat, 'black', False)

        '''
              获取区分度矩阵
              '''
        d_mat = dd.from_originmat_to_dmat(gray)
        # can_blur =  cv2.Canny(blur_res, 50, 120)
        bool_repair = False
        '''
         修复噪声之后的mat
         '''
        repair_mat = rn.repair_noise(gray)

        tl.after_repair_min_max_inter_max_to_excel('repaired', gray, ret_mat, inter_mat, repair_mat)
        origin_gray = gray
        gray = repair_mat
        bool_repair = True

        # 207 213 338 344
        '''
              获取修复噪声后区分度矩阵
        '''
        repair_d_mat = dd.from_originmat_to_dmat(gray)

        '''
        给图像添加噪声
        '''
        gray = an.addSaltNoise(gray, 0.99)
        # cv2.imshow('origin_salt_noise', gray)

        # gray = an.GaussianNoise(gray, 0, 0.1, 8)
        # cv2.imshow('origin_gausssian_noise', gray)

        can = cv2.Canny(gray, 50, 120)
        h, w = gray.shape

        '''
        获取groundtruth
        '''
        ground_mat = self.get_groundtruth_mat()


        # print('max', max(inter_mat.flatten()))
        # print('min', min(inter_mat.flatten()))



        # 矛盾矩阵
        # md_mat = cm.from_gray_to_md_mat(gray)
        # cv2.imshow('md_mat', md_mat)
        # print(h, w, md_mat.shape)
        # print(md_mat)


        #



        # cv2.imshow('d_mat', d_mat)
        '''
        修复矛盾点
        '''
        # gray = rmd.fix_md(gray)

        # 叉边方法
        bin = dd.link_edge_all_mat_by_chabian(gray, 10)
        bin = self.repair_bin(bin)
        tl.drawBinPicByChabian('my_algo', bin)

        '''
        F seasure
        '''
        my_bin = self.adjust_0_255(bin);
        _, _, f_score = ms.F(ground_mat, my_bin)
        # print('\nf_score= ', f_score)

        # bin = dd.link_edge_all_mat_by_chabian_v1(gray, 15)
        if bool_repair == False:
            tl.gray_dmat_big_small_to_excel('big_small_edge', gray, d_mat, bin, repair_d_mat)
        else:
            tl.after_repair_show_diff_to_excel('repaired_show_diff', origin_gray, ret_mat, d_mat, repair_mat, repair_d_mat, bin)

        # bin_blur = dd.link_edge_all_mat_by_chabian(blur_res, 15)
        # bin_blur = self.repair_bin(bin_blur)
        # tl.drawBinPic('new_blur', bin)

        self.show_rgb(bin, can, inter_mat)
        self.show_canny_and_origin_rgb(can, 'black', False)
        self.show_myalgo_and_origin_pic_rgb(bin, 'black', False)
        '''
        显示
        '''
        # self.single_pic_show_rgb(inter_mat)# 只显示干扰能

        cv2.imshow('canny', can)

        # cv2.imwrite('./DestImage/leaf.jpg', tmp)
        cv2.waitKey(0)


        # x, y = 0, 0
        # cnt = 0
        # while True:
        #     x, y = dd.find_local_max__(x, y, gray, d_mat, 10)
        #     if (x==-1 and y==-1):
        #         break
        #     print(x,'\t', y, '\t', gray[x, y])
        #
        #     y += 1
        #     cnt += 1
        # tl.gray_dmat_to_excel1('gray_dmat', gray, d_mat)
        # print('cnt', cnt)

        # 对区分度画直方图
        # d_list = d_mat.flatten()
        # tl.drawHist(d_list, "dis_val", "number")

        # e_mat = dd.distinction_degree_elimination_allmat_before(gray)

        # sho_mat = dd.get_sho_mat(gray)
        # # 对阈值画直方图
        # sho_list = sho_mat.flatten()
        # tl.drawHist(sho_list,"sho_val", "number")

        # mark_mat = np.zeros(gray.shape, dtype=np.uint8)
        # ground_mat = self.get_groundtruth_mat()
        # h, w = gray.shape
        # for i in range(h):
        #     for j in range(w):
        #         # if d_mat[i, j] > sho and sho_mat[i, j] > sho:
        #         if d_mat[i, j] > sho:
        #             mark_mat[i, j] = 1
        # mark_mat = mark_mat & e_mat
        # tl.gray_dmat_mark_to_excel('gray_dif_sho_new_think-5-24', gray, d_mat,mark_mat, ground_mat)
        # tl.drawBinPic('new_think2_e_mat', e_mat)
        # tl.drawBinPic('new_think2_mark_mat', mark_mat)

    '''
    只去掉区分度最小的那个
    '''
    def new_think3(self, sho):
        gray = self.get_gray()
        d_mat = dd.from_originmat_to_dmat(gray)
        # sho_mat = dd.get_sho_mat(gray)
        mark_mat = np.zeros(gray.shape, dtype=np.uint8)
        ground_mat = self.get_groundtruth_mat()
        h, w = gray.shape
        for i in range(h):
            for j in range(w):
                # if d_mat[i, j] > sho and d_mat[i, j] > sho_mat[i, j]:
                if d_mat[i, j] > sho:
                    mark_mat[i, j] = 255

        cv2.imshow('new_think', mark_mat)
        tl.gray_dmat_mark_to_excel('gray_dif_new_think', gray, d_mat, mark_mat, ground_mat)


    def center_sho_5_13_test(self):
        h, w = self.mat.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                dd.center_sho_5_13(i, j, self.mat)

    def twoDataHist(self):
        gray = self.get_gray()
        d_mat = dd.from_originmat_to_dmat(gray)
        sho_mat = dd.get_sho_mat(gray)
        dl = d_mat.flatten().tolist()
        sl = sho_mat.flatten().tolist()
        # test = pd.DataFrame([dl,
        #                      sl])
        # plt.hist(test.values.T)
        # plt.show()

        import numpy as np
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-deep')

        # x = np.random.normal(1, 2, 5000)
        # y = np.random.normal(-1, 3, 2000)
        max_ = max(max(dl), max(sl))
        bins = np.linspace(0, max_, max_)

        plt.hist([dl, sl], bins, label=['dl', 'sl'])
        plt.legend(loc='upper right')
        plt.show()



    # def pic_show_5_28(self):
    #     gray = [
    #         [117,118,119,119,119,119,119],
    #         [119,118,119,121,119,120,122],
    #         [120,124,113,114,120,120,120],
    #         [119,124,110,118,125,120,122],
    #         [120,123,105,116,127,118,115],
    #         [123,120,101,109,111,87,78],
    #         [98,93,81,91,100,92,92],
    #         [92,94,92,97,96,92,95],
    #         [96,98,94,92,90,91,89],
    #     ]
    #     groundTruth = [
    #         [0,0,0,0,0,0,0],
    #         [0,0,0,0,0,0,0],
    #         [0,0,0,0,1,1,1],
    #         [1,1,1,1,0,0,0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #     bin_big = [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 1, 0, 0, 0],
    #         [0, 1, 0, 0, 1, 0, 0],
    #         [0, 1, 0, 0, 1, 0, 0],
    #         [0, 1, 0, 0, 1, 1, 1],
    #         [1, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #     bin_small = [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 1, 0, 1, 1, 1],
    #         [1, 1, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #
    #     bin_small =np.array(bin_small, dtype=np.uint8)
    #     bin_big =np.array(bin_big, dtype=np.uint8)
    #     gt =np.array(groundTruth, dtype=np.uint8)
    #     gray = np.array(gray, dtype=np.uint8)
    #
    #     bin_b = np.zeros(gray.shape, dtype=np.uint8)
    #     bin_s = np.zeros(gray.shape, dtype=np.uint8)
    #     bin_gt = np.zeros(gray.shape, dtype=np.uint8)
    #     h, w = gray.shape
    #     for i in range(h):
    #         for j in range(w):
    #             if bin_big[i,j] > 0:
    #                 bin_b[i,j] = 255
    #             if gt[i,j] > 0:
    #                 bin_gt[i,j] = 255
    #             if bin_small[i,j]>0:
    #                 bin_s[i,j] = 255
    #
    #
    #
    #     cv2.imshow('m5-28', gray)
    #     cv2.imwrite('./gray_edge_gt.jpg', gray)
    #     cv2.imwrite('./bin_b.jpg', bin_b)
    #     cv2.imwrite('./bin_s.jpg', bin_s)
    #
    #     cv2.imwrite('./gt.jpg', bin_gt)
    #
    #
    # def pic_show_5_28_2(self):
    #     gray = [
    #        [29,32,30,31,32,27,24],
    #         [28,26,33,35,31,31,31],
    #         [26,31,27,28,33,33,30],
    #         [33,25,22,21,24,25,38],
    #         [139,120,106,96,104,123,147],
    #         [155,156,153,157,161,155,152],
    #         [143,143,147,144,140,146,148],
    #         [147,145,143,143,144,144,146],
    #         [144,143,144,145,145,143,144]
    #     ]
    #     groundTruth = [
    #         [0,0,0,0,0,0,0],
    #         [0,0,0,0,0,0,0],
    #         [0,0,0,0,1,1,1],
    #         [1,1,1,1,0,0,0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #     bin_big = [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #     bin_small = [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    #
    #     bin_small =np.array(bin_small, dtype=np.uint8)
    #     bin_big =np.array(bin_big, dtype=np.uint8)
    #     gt =np.array(groundTruth, dtype=np.uint8)
    #     gray = np.array(gray, dtype=np.uint8)
    #
    #     bin_b = np.zeros(gray.shape, dtype=np.uint8)
    #     bin_s = np.zeros(gray.shape, dtype=np.uint8)
    #     bin_gt = np.zeros(gray.shape, dtype=np.uint8)
    #     h, w = gray.shape
    #     for i in range(h):
    #         for j in range(w):
    #             if bin_big[i,j] > 0:
    #                 bin_b[i,j] = 255
    #             if gt[i,j] > 0:
    #                 bin_gt[i,j] = 255
    #             if bin_small[i,j]>0:
    #                 bin_s[i,j] = 255
    #
    #
    #
    #     cv2.imshow('m5-28', gray)
    #     cv2.imwrite('./gray_edge_gt_2.jpg', gray)
    #     cv2.imwrite('./bin_b_2.jpg', bin_b)
    #     cv2.imwrite('./bin_s_2.jpg', bin_s)
    #
    #     cv2.imwrite('./gt_2.jpg', bin_gt)

    '''
    @:param 从(sx, sy)开始搜索局部极大值
    @:return  返回一个局部极大值坐标
    '''
    def find_local_max__(self, sx, sy, gray, d_mat, bin_mat = [], sho = 0):
        # 找局部极大值点
        h, w = gray.shape
        for i in range(sx, h):
            for j in range(sy, w):
                if d_mat[i,j] >= sho:
                    isT = True
                    for i_ in range(i-1, i+2):
                        if isT == False:
                            break
                        for j_ in range(j-1, j+2):
                            if (i!=i_ and j!=j_) and d_mat[i,j] < d_mat[i_, j_]:
                                isT = False
                                break
                    if isT == True:
                        # self.start_with(i, j, gray, d_mat, e_box)
                        return i, j

        return -1, -1

    def link_edge_new_method_5_29(self, gray, sho = 0):
        d_mat = dd.from_originmat_to_dmat(gray)
        bin_mat = np.zeros(gray.shape, dtype=np.uint8)
        s_x, s_y = 0, 0
        s_x, s_y = self.find_local_max__(s_x, s_y, gray, d_mat, bin_mat, sho)
        end_p, edge_p = (), 0
        flag = False
        while flag == False:
            flag = self.start_with(s_x, s_y, gray, d_mat, end_p, edge_p)
            if flag == True:
                break
            else:
                s_x, s_y = self.find_local_max__(s_x, s_y, gray, d_mat, bin_mat, sho)

        # 此时以点end_p, 边属性edge_p 开始延伸
        r_bool = True
        while r_bool == True:
            r_bool = self.expand_by_end_p(gray, d_mat, e_box, bin_mat, end_p, edge_p)

        return bin_mat

    # 以end_p 和 edge_p 来更新bin_mat
    def expand_by_end_p(self, gray, d_mat, e_box, bin_mat, end_p, edge_p):
        i, j = end_p[0], end_p[1]

        lu_coor = [(i-1,j-1), (i-1, j), (i,j-1), (i,j)]
        ru_coor = [(i-1,j), (i-1, j+1), (i,j), (i,j+1)]
        ld_coor = [(i, j-1), (i,j), (i+1,j-1), (i+1,j)]
        rd_coor = [(i,j), (i,j+1), (i+1,j),(i+1,j+1)]
        coor_list = [lu_coor, ru_coor, ld_coor, rd_coor]

        x, y = 0, 0
        mark_list = [(x, y), (x, y+2), (x+2, y), (x+2, y+2)]

        left_up = gray[i-1:i+1, j-1:j+1]
        right_up = gray[i-1:i+2, j:j+2]
        left_down = gray[i:i+2, j-1:j+2]
        right_down = gray[i:i+2, j:j+2]
        mat2_list = [left_up, right_up, left_down, right_down]

        box_coor_list = [(i-1,j-1), (i-1,j), (i,j), (i,j-1)]
        same_p = [] # 点属性相同的点坐标
        r_bool = False
        for ind in range(4):
            # 如果当前格子被消去，直接跳过此格子
            if e_box[box_coor_list[ind]] == 0:
                continue

            f = True
            for (i_, j_) in coor_list[ind]:
                if bin_mat[i_, j_] == 1:
                    f = False
                    break

            if f == True:
                r_bool = True
                fb_3 = cm.fenbian__(mat2_list[ind])
                if cm.check4__(0, 0, fb_3) == True:
                    for coor_ in coor_list[ind]:
                        same_p.append(coor_)
                elif cm.check3__(0, 0, fb_3) == True:
                    mark_3 = cm.mark_bsm3__(0, 0, fb_3)
                    for i_ in range(4):
                        if edge_p == mark_3[mark_list[i_]]:
                            same_p.append(coor_list[ind][i_])
                elif cm.check2__(0, 0, fb_3) == True:
                    mark_3 = cm.mark_bsm2__(0, 0, fb_3)
                    for i_ in range(4):
                        if edge_p == mark_3[mark_list[i_]]:
                            same_p.append(coor_list[ind][i_])
        if r_bool == False or same_p == []:
            return False
        d_mat_list = []
        for i_ in same_p:
            d_mat_list.append(-d_mat(i_))

        sort_list = np.argsort(d_mat_list)

        if len(sort_list) == 1:
            end_p = d_mat_list[sort_list[0]]

        else:
            # 判断是否有两个区分度值相同
            if d_mat[d_mat_list[sort_list[0]]] == d_mat[d_mat_list[sort_list[1]]]:
                if abs(gray[d_mat_list[sort_list[0]]] - gray[end_p]) <abs(gray[d_mat_list[sort_list[1]]] - gray[end_p]):
                        end_p = d_mat_list[sort_list[0]]
                else:
                    end_p = d_mat_list[sort_list[1]]

        return True











    def start_with(self, s_x, s_y, gray, d_mat, end_p, edge_p):
        # 判断起点格子边的属性
        mat_2 = gray[s_x : s_x + 2, s_y : s_y + 2]
        fb_3 = cm.fenbian__(mat_2, 0, 0)
        if cm.check4__(0, 0, fb_3) == True:
            return False

        if cm.check2__(0, 0, fb_3) == True:
            mark_mat3 = cm.mark_bsm2__(0, 0, fb_3)
            if mark_mat3[s_x, s_y+1] == mark_mat3[s_x, s_y]:
                end_p = (s_x, s_y+1)
                edge_p = mark_mat3[s_x, s_y]
                return True
            if mark_mat3[s_x+1, s_y] == mark_mat3[s_x, s_y]:
                end_p = (s_x+1, s_y)
                edge_p = mark_mat3[s_x, s_y]
                return True

        if cm.check3__(0, 0, fb_3) == True:
            mark_mat3 = cm.mark_bsm3__(0, 0, fb_3)
            # 5 5
            # 5 6
            if mark_mat3[s_x, s_y] != mark_mat3[s_x+1, s_y+1]:
                if d_mat[s_x, s_y+1] >= d_mat[s_x+1, s_y]: #
                    end_p = (s_x, s_y+1)
                    edge_p = mark_mat3[s_x, s_y]
                    return True
                else:
                    end_p = (s_x+1, s_y)
                    edge_p = mark_mat3[s_x, s_y]
                    return True
            # 5 5
            # 6 5
            else:
                end_p = (s_x + 1, s_y + 1)
                edge_p = mark_mat3[s_x, s_y]
                return True


test = DDTest()
# test.remove_extra_all_mat_test()
# test.center_sho_5_13_test()
# test.new_think3(25)

# test.new_think2(11)

# test.show_diff(10)
# test.twoDataHist()
# test.pic_show_5_28_2()


cv2.waitKey(0)


