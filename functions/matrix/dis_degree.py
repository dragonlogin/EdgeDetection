import numpy as np

from functions.matrix.check_mat import check_mat
from tools.tools import tools

# from Function.Interference.Interference import Interference
# inter = Interference()
cm = check_mat()
tl = tools()


class dis_degree(object):
    '''
    :param 传入一个原始mat2， 大小边mat3
    :return 一个区分度值
    '''

    def pro_diff_edge__(self, mat_2, mat_3):
        pl2 = [(0, 0), (0, 1), (1, 1), (1, 0)]
        pl3 = [(0, 0), (0, 2), (2, 2), (2, 0)]
        sl, bl = [], []

        for i in range(4):
            if mat_3[pl3[i]] == 5:
                sl.append(mat_2[pl2[i]])
            elif mat_3[pl3[i]] == 6:
                bl.append(mat_2[pl2[i]])

        s_max = max(sl)
        b_min = min(bl)
        dif_ret = abs(int(s_max) - int(b_min))
        # print('dif_ret\n', dif_ret)
        return dif_ret

    '''
    :param 传入一个2x2mat
    :return  返回一个区分度值, 错误返回 -1 ， 成功返回 区分度值
    '''
    def diff_edge_2x2__(self, mat_2) -> int:
        # print("\nmat_2=\n{}".format(mat_2))
        edge_mat_3 = cm.fenbian__(mat_2)
        # print("\nmat_3=\n{}".format(edge_mat_3))
        if cm.check4__(0, 0, edge_mat_3):
            return 0
        else:
            if cm.check2__(0, 0, edge_mat_3):
                mark_mat_3 = cm.mark_bsm2__(0, 0, edge_mat_3)
                dif_ret = self.pro_diff_edge__(mat_2, mark_mat_3)
                return dif_ret

            elif cm.check3__(0, 0, edge_mat_3):
                mark_mat_3 = cm.mark_bsm3__(0, 0, edge_mat_3)
                dif_ret = self.pro_diff_edge__(mat_2, mark_mat_3)
                return dif_ret

    '''
    para: 传入一个原始mat
    return： 返回一个经过规则处理的区分度mat
    规则为：阈值大于区分度值，则消去
    '''
    def dif_degree_mat(self, in_mat, d_mat):
        h, w = in_mat.shape
        # dd_mat = np.ones(in_mat.shape, dtype = np.uint8)
        dd_mat = d_mat.copy()
        # 处理最后一行和最后一列
        # for j in range(w):
        #     dd_mat[h - 1, j] = 0
        # for i in range(h):
        #     dd_mat[i, w - 1] = 0

        sho_mat = self.get_sho_mat(in_mat)
        # dv_mat = np.zeros(in_mat.shape, dtype = np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):

                # lu_mat2 = in_mat[i - 1 : i + 1, j - 1 : j + 1]
                # ru_mat2 = in_mat[i - 1 : i + 1, j : j + 2]
                # ld_mat2 = in_mat[i : i + 2, j - 1 : j + 1]
                # rd_mat2 = in_mat[i : i + 2, j : j + 2]
                # # mat2 = in_mat[i : i + 2, j : j + 2]
                #
                # lu_dif = self.diff_edge_2x2__(lu_mat2)
                # ru_dif = self.diff_edge_2x2__(ru_mat2)
                # ld_dif = self.diff_edge_2x2__(ld_mat2)
                # rd_dif = self.diff_edge_2x2__(rd_mat2)

                # if lu_dif != -1:
                #     dv_mat[i - 1, j - 1] = lu_dif

                pl = [(i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)]
                # dl = [lu_dif, ru_dif, ld_dif, rd_dif]

                # for k in range(4):
                #     if dd_mat[pl[k]] == 0 or dl[k] == 0 or  dl[k] <= sho_mat[i, j]:
                #         dd_mat[pl[k]] = 0
                #     else:
                #         dd_mat[pl[k]] = dl[k]

                for k in range(4):
                    if dd_mat[pl[k]] == 0 or dd_mat[pl[k]] <= sho_mat[i, j]:
                        dd_mat[pl[k]] = 0
        # return dd_mat, dv_mat
        return dd_mat

    def combine_e_d(self, e_mat_before, d_mat):
        h, w = e_mat_before.shape
        dd_mat = e_mat_before.copy()
        for i in range(h):
            for j in range(w):
                if d_mat[i, j] == 0:
                    dd_mat[i, j] = 0

        return dd_mat

    # test
    def dif_degree_mat_test(self, gray):
        mat = self.dif_degree_mat(gray)
        tl.singleAndGreaterThanZero('dif_mat', gray, mat)

        print('dif_mat\n', mat)

    def proDifDegree(self, gray):
        h, w = gray.shape
        re_mat = np.ones(gray.shape, dtype=np.uint8)
        dif_mat = self.dif_degree_mat(gray)

        for i in range(h - 1):
            for j in range(w - 1):
                mat2 = dif_mat[i:i + 2, j:j + 2]

    '''
    @:param 传入灰度图像
    @:return 返回消去区分度值矩阵
    '''

    def distinction_degree_elimination_allmat_after(self, gray, dd_mat):
        h, w = gray.shape
        e_mat = np.ones(gray.shape, dtype=np.uint8)
        # 区分度值矩阵
        # d_mat, dv_mat = self.dif_degree_mat(gray)
        d_mat = self.dif_degree_mat(gray, dd_mat)

        # for i in range(h - 1):
        #     for j in range(w - 1):
        #         d_mat2 = d_mat[i : i + 2, j : j + 2]
        #         e_mat2 = self.distinction_degree_elimination_mat2(d_mat2)
        #
        #         for ii in range(2):
        #             for jj in range(2):
        #                 if e_mat[i + ii, j + jj] == 1:
        #                     e_mat[i + ii, j + jj] = e_mat2[ii, jj]

        for i in range(h):
            for j in range(w):
                if d_mat[i, j] == 0:
                    e_mat[i, j] = 0
        # return d_mat, e_mat, dv_mat
        return e_mat

    '''
    @:param 传入一个-原来矩阵o_mat
    @:return 返回一个区分度矩阵d_mat
    '''

    def from_originmat_to_dmat(self, o_mat):
        h, w = o_mat.shape
        d_mat = np.zeros(o_mat.shape, dtype=np.uint8)

        for i in range(h - 1):
            for j in range(w - 1):
                mat2 = o_mat[i: i + 2, j: j + 2]
                dif_val = self.diff_edge_2x2__(mat2)
                d_mat[i, j] = dif_val

        return d_mat

    def distinction_degree_elimination_allmat_before(self, gray):
        h, w = gray.shape
        e_mat = np.ones(gray.shape, dtype=np.uint8)
        # 区分度值矩阵
        # _, dv_mat = self.dif_degree_mat(gray)
        dv_mat = self.from_originmat_to_dmat(gray)

        for i in range(h - 1):
            for j in range(w - 1):
                d_mat2 = dv_mat[i: i + 2, j: j + 2]
                e_mat2x2 = e_mat[i: i + 2, j: j + 2]
                e_mat2 = self.distinction_degree_elimination_mat2(d_mat2, e_mat2x2)

                for ii in range(2):
                    for jj in range(2):
                        if e_mat[i + ii, j + jj] == 1:
                            e_mat[i + ii, j + jj] = e_mat2[ii, jj]

        return e_mat

    # test
    def d_d_e_a_test(self, gray):
        d_mat, e_mat = self.distinction_degree_elimination_allmat(gray)

        print('\ne_mat\n', e_mat)
        tl.singleAndGreaterThanZero('e_mat', d_mat, e_mat)

    # '''
    # 5.25修改，改为直接删除2个区分度值最小的2个
    # '''
    # def distinction_degree_elimination_mat2(self, d_mat2):
    #     h, w = d_mat2.shape
    #     e_mat2 = np.ones(d_mat2.shape, np.uint8).flatten()
    #     d_mat2 = d_mat2.flatten()
    #     sort_i = np.argsort(d_mat2)
    #     pass

    '''
    @:param 传入一个 2x2 区分度值 mat
    @:return 返回一个 2x2 01 消去 mat， 0代表消去
    '''

    def distinction_degree_elimination_mat2(self, d_mat2, e_mat2x2):
        h, w = d_mat2.shape
        e_mat2 = np.ones(d_mat2.shape, np.uint8).flatten()
        d_mat2 = d_mat2.flatten()
        e_mat2x2 = e_mat2x2.flatten()

        # print('d_mat2', d_mat2)

        sort_i = np.argsort(d_mat2)
        # print('sort_i', sort_i)

        d_mat2_list = d_mat2.tolist()
        s_set = set(d_mat2_list)

        lens, lenm = len(s_set), len(d_mat2)
        # 没有相同值
        if lens == lenm:
            e_mat2[sort_i[0]] = e_mat2[sort_i[1]] = 0
        # 有 2 个相同值
        elif lens + 1 == lenm:
            # A = A > B > C
            if d_mat2[sort_i[2]] == d_mat2[sort_i[3]]:
                # e_mat2[sort_i[0]] = e_mat2[sort_i[1]] = 0
                if e_mat2x2[sort_i[2]] == 1 and e_mat2x2[sort_i[3]] == 1:
                    e_mat2[sort_i[2]] = 0
                e_mat2[sort_i[0]] = 0
            # A > B1 = B2 > C
            elif d_mat2[sort_i[1]] == d_mat2[sort_i[2]]:
                # 如果 B1 和 B2都没去掉，则随便去掉一个，否则去掉最小的和已经去掉的B1或者B2
                if e_mat2x2[sort_i[1]] == 1 and e_mat2x2[sort_i[2]] == 1:
                    e_mat2[sort_i[1]] = 0
                e_mat2[sort_i[0]] = 0

                # A > B > C = C
            elif d_mat2[sort_i[0]] == d_mat2[sort_i[1]]:
                if e_mat2x2[sort_i[0]] == 1 and e_mat2x2[sort_i[1]] == 1:
                    e_mat2[sort_i[0]] = 0

            # e_mat2[sort_i[0]] = 0
        # 有 3 个相同值
        elif lens + 2 == lenm:
            # if d_mat2[sort_i[1]] == d_mat2[sort_i[2]] == d_mat2[sort_i[3]]:
            #     e_mat2[sort_i[0]] = 0
            #
            # elif d_mat2[sort_i[1]] == d_mat2[sort_i[2]] == d_mat2[sort_i[0]]:
            #     if e_mat2[sort_i[0]] == 1 and e_mat2[sort_i[1]] == 1 and e_mat2[sort_i[2]] == 1:
            #         e_mat2[sort_i[0]] = 0

            e_mat2[sort_i[0]] = 0
        # 有 4 个相同值
        elif lens == 1:
            if d_mat2[sort_i[0]] == 0:
                e_mat2[sort_i[0]] = e_mat2[sort_i[1]] = e_mat2[sort_i[2]] = e_mat2[sort_i[3]] = 0

        e_mat2 = e_mat2.reshape([2, 2])
        # print('e_mat2\n', e_mat2)
        return e_mat2

    def remove_extra_5x5(self, d_mat, mark, lx, ly, len=5):
        sum, d_sz = 0, 0
        m_sz = 0

        for i in range(lx, lx + len):
            for j in range(ly, ly + len):
                if d_mat[i, j] > 0:
                    # if mark[i, j] == 1:
                    sum += d_mat[i, j]
                    d_sz += 1
                if mark[i, j] == 1:
                    m_sz += 1

        if d_sz == 0 or m_sz == 0:
            return

        avg = sum / d_sz
        for i in range(lx, lx + len):
            for j in range(ly, ly + len):
                if mark[i, j] == 1 and avg >= d_mat[i, j]:
                    mark[i, j] = 0

    def remove_extra_all_mat(self, gray, len=5):
        d_mat, mark, dv_mat = self.distinction_degree_elimination_allmat_after(gray)
        h, w = gray.shape

        # for i in range(h - len):
        #     for j in range(w - len):
        #         self.remove_extra_5x5(d_mat, mark, i, j, len)

        return d_mat, mark, dv_mat

    '''
    @:param 传入一个mat2
    @:return 返回一个最大差
    '''

    def max_difference_mat2_5_13(self, mat2):
        fb_mat3 = cm.fenbian__(mat2)
        # print('mat2\n', mat2)
        # print('fb_mat3\n', fb_mat3)
        lx, ly = 0, 0
        max_dif_val = 0
        sei, bei = cm.bigedge_smalledge_index(fb_mat3)
        se = [mat2[i] for i in sei]
        be = [mat2[i] for i in bei]
        # print('se', se)
        # print('be', be)

        # 同一区域
        if se == [] and be == []:
            mat2_f = mat2.flatten()
            max_dif_val = max(mat2_f) - min(mat2_f)

        else:
            s_max_dif = max(se) - min(se)
            b_max_dif = max(be) - min(be)
            max_dif_val = max(s_max_dif, b_max_dif)

        # print('max_dif_val:', max_dif_val)

        return max_dif_val

    '''
    @:param mat2
    @:return bool
    返回是否同一区域，如果
    0:4 ->true
    1:3 return true if can merge, otherwise, return false
    2:2 return true if can merge, otherwise, return false
    merge operation is meaning that the maximum difference of The same area greater than distinguish value
    '''

    def whether_merge(self, mat2) -> bool:
        fb_mat3 = cm.fenbian__(mat2)

        if cm.check4__(0, 0, fb_mat3) == True:
            return True

        sei, bei = cm.bigedge_smalledge_index(fb_mat3)
        se = [mat2[i] for i in sei]
        be = [mat2[i] for i in bei]

        if cm.check3__(0, 0, fb_mat3) == True:
            if len(se) == 3:
                same_maxd = max(se) - min(se)
                d_val = be[0] - max(se)
            else:
                same_maxd = max(be) - min(be)
                d_val = se[0] - max(be)

            return same_maxd > d_val

        if cm.check2__(0, 0, fb_mat3) == True:
            same_maxd = max(max(se) - min(se), max(be) - min(be))
            d_val = min(be) - min(se)
            return same_maxd > d_val

    '''
    @:param 传入一个中心点 (x, y), 传入的中心点已经满足要求
    @:return 返回一个中心点的阈值
    '''

    def center_sho_5_13(self, cx, cy, mat):
        sho = -1
        lu_mat2 = mat[cx - 1: cx + 1, cy - 1: cy + 1]
        ru_mat2 = mat[cx - 1: cx + 1, cy: cy + 2]
        ld_mat2 = mat[cx: cx + 2, cy - 1: cy + 1]
        rd_mat2 = mat[cx: cx + 2, cy: cy + 2]

        sho = max(sho, self.max_difference_mat2_5_13(lu_mat2))
        sho = max(sho, self.max_difference_mat2_5_13(ru_mat2))
        sho = max(sho, self.max_difference_mat2_5_13(ld_mat2))
        sho = max(sho, self.max_difference_mat2_5_13(rd_mat2))

        # print('sho', sho)
        return sho

    '''
    @:param 传入一个原始mat
    @:return 返回一个阈值mat
    '''

    def get_sho_mat(self, in_mat):
        h, w = in_mat.shape
        sho_mat = np.zeros(in_mat.shape, dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sho = self.center_sho_5_13(i, j, in_mat)
                sho_mat[i, j] = sho

        return sho_mat

    '''
    @param 传入一个2x2mat
    @return 返回一个3x3消去无用边的mat
    发现跟最开始的分边是一样的。
    '''

    def eli_useless_edge_2x2(self, mat2):
        h, w = mat2.shape
        # e_mat3 = np.ones((h*2-1, w*2-1), dtype=np.uint8)
        fb_mat3 = cm.fenbian__(mat2)

        # 如果可以合并
        if self.whether_merge(mat2) == True:
            e_mat3 = np.zeros((h * 2 - 1, w * 2 - 1), dtype=np.uint8)
            for i in range(2):
                for j in range(2):
                    e_mat3[i * 2, j * 2] = mat2[i, j]
            return e_mat3
        # print('\nmat2\n', mat2)
        # print('fb_mat3\n', fb_mat3)
        # print('mark_edge_3\n', cm.mark_bsm3__(0, 0, fb_mat3))
        return fb_mat3

    def emat_conver_to_bin(self, e_mat):
        h, w = e_mat.shape
        bin = np.zeros((h, w), dtype=np.uint8)

        # process rows
        for i in range(1, h, 2):
            for j in range(0, w, 2):
                if e_mat[i, j] > 0:
                    bin[i, j] = 1

        # process cols
        for i in range(0, h, 2):
            for j in range(1, w, 2):
                if e_mat[i, j] > 0:
                    bin[i, j] = 1

        # precess duijiao
        for i in range(1, h, 2):
            for j in range(1, w, 2):
                if e_mat[i, j] > 0:
                    bin[i, j] = 1

        return bin

    def eli_useless_edge_allmat(self, mat):
        h, w = mat.shape
        e_mat = np.ones((h * 2 - 1, w * 2 - 1), dtype=np.uint8)
        # bin = np.zeros((h*2-1, w*2-1), dtype=np.uint8)

        for i in range(h - 1):
            for j in range(w - 1):
                mat2 = mat[i: i + 2, j: j + 2]
                e_mat3 = self.eli_useless_edge_2x2(mat2)
                # print('e_mat3\n', e_mat3)

                # 进行覆盖, 这里涉及2x2 与 3x3 坐标的转换
                #

                i3, j3 = i * 2, j * 2
                # o_mat3 = e_mat[i3 : i3 + 3, j3 : j3 + 3]
                # if sum(e_mat3.flatten()) == 0:
                #     continue
                # else:
                #     if sum(o_mat3.flatten()) != 0:
                #         e_mat3 = e_mat3 & o_mat3

                for i_ in range(3):
                    for j_ in range(3):
                        if e_mat[i_ + i3, j_ + j3] == 1:
                            e_mat[i_ + i3, j_ + j3] = e_mat3[i_, j_]

                # print('e_mat\n', e_mat)

        return e_mat

    # test
    def eli_useless_edge_allmat_test(self, mat):
        e_mat = self.eli_useless_edge_allmat(mat)
        print('e_mat\n', e_mat)
        bin = self.emat_conver_to_bin(e_mat)
        tl.singleAndGreaterThanZero('6-1', e_mat, bin)

    # test
    def eli_useless_edge_test(self, mat):
        h, w = mat.shape
        for i in range(h - 1):
            for j in range(w - 1):
                mat2 = mat[i: i + 2, j: j + 2]
                self.eli_useless_edge_2x2(mat2)

        print('eli_useless_edge_test ok')

    '''
    @:param 从(sx, sy)开始搜索局部极大值
    @:return  返回一个局部极大值坐标
    '''

    def find_local_max__(self, sx, sy, gray, d_mat, sho=0, num=2, bin_mat=[]):
        # 找局部极大值点
        h, w = gray.shape
        for i in range(sx, h):
            for j in range(sy, w):
                if d_mat[i, j] >= sho and (i - 1 >= 0 and i + 1 < h and j - 1 >= 0 and j + 1 < w):
                    # cnt = 0
                    # isT = True
                    # for i_ in range(-1, 2):
                    #     if isT == False:
                    #         break
                    #     for j_ in range(-1, 2):
                    #         dx, dy = i_ + i, j_ + j
                    #         if (0==i_ and 0==j_) or dx<0 or dx>=h or dy<0 or dy>=w:
                    #             continue
                    #         if d_mat[i, j] < d_mat[dx, dy]:
                    #             cnt+=1
                    #             isT = False
                    #             break
                    # if cnt<= 0:
                    #     isT = True
                    # if isT == True:
                    #     # self.start_with(i, j, gray, d_mat, e_box)
                    #     return i, j
                    #

                    mat3 = d_mat[i - 1: i + 2, j - 1: j + 2]
                    mat3 = mat3.flatten()
                    cen_val = mat3[4]

                    # mat3 = np.array(list(set(mat3)))
                    # print('mat3', mat3)
                    index = np.argsort(mat3)
                    # print('index', index)
                    # c = [mat3[index[i]] for i in range(9)]
                    # print('c', c)
                    isT = False
                    for k in range(-num, 0):
                        if mat3[index[k]] == cen_val:  # 4
                            isT = True
                            break

                    if isT == True:
                        return i, j

                if j + 1 == w:
                    sy = 0

        return -1, -1

    '''
        新的分边函数，加入了一个新的规则：
        如果同一区域的最大差大于区分度值，则当成同一区域来处理。
        '''

    def new_fenbian__(self, mat2, x=0, y=0):
        h, w = mat2.shape
        fb_mat3 = np.zeros((h * 2 - 1, w * 2 - 1), dtype=np.uint8)
        if self.whether_merge(mat2):
            return fb_mat3
        else:
            return cm.fenbian__(mat2)

    def cha_bian_2x2__(self, lx, ly, mat):
        h, w = mat.shape
        st = set()
        if lx + 1 >= h or ly + 1 >= w:
            return st

        mat2 = mat[lx: lx + 2, ly: ly + 2]

        # 最大差大于区分度值未合并的写法
        fb_mat3 = cm.fenbian__(mat2)

        # 合并的写法 经测试，不合并效果更好
        # fb_mat3 = self.new_fenbian__(mat2)

        # print('mat2\n', mat2)
        # print('fb_mat3\n', fb_mat3)

        p_2 = [(lx, ly), (lx, ly + 1), (lx + 1, ly + 1), (lx + 1, ly)]
        x, y = 0, 0
        e_3 = [(x, y + 1), (x + 1, y + 2), (x + 2, y + 1), (x + 1, y)]
        cb = []  # 保存叉边的序号

        if cm.check4__(0, 0, fb_mat3):
            return st
        else:
            if fb_mat3[e_3[0]] == 0 and fb_mat3[e_3[1]] == 0:
                cb.extend([0, 1])
            elif fb_mat3[e_3[0]] == 0 and fb_mat3[e_3[2]] == 0:
                cb.extend([0, 2])
            elif fb_mat3[e_3[0]] == 0 and fb_mat3[e_3[3]] == 0:
                cb.extend([0, 3])
            elif fb_mat3[e_3[1]] == 0 and fb_mat3[e_3[2]] == 0:
                cb.extend([1, 2])
            elif fb_mat3[e_3[1]] == 0 and fb_mat3[e_3[3]] == 0:
                cb.extend([1, 3])
            elif fb_mat3[e_3[2]] == 0 and fb_mat3[e_3[3]] == 0:
                cb.extend([2, 3])
        # debug
        # print('cb', cb, (lx, ly))
        for i in cb:
            if i == 2:
                val = (p_2[3], p_2[2])
            elif i == 3:
                val = (p_2[0], p_2[3])
            else:
                val = (p_2[i], p_2[(i + 1) % 4])
            st.add(val)
        return st

    '''
    追踪算法
    @:param 
    '''

    def start_with_point(self, sx, sy, mat, d_mat, sho=0, mark_st=set()):
        from queue import Queue
        h, w = mat.shape
        x_end, y_end = sx, sy

        if (sx, sy) in mark_st:
            return [], (sx, sy + 1)

        st = self.cha_bian_2x2__(sx, sy, mat)
        save_st = set()
        q = Queue()
        for i in st:
            save_st.add(i)
            q.put(i)

        while q.qsize() > 0:
            p = q.get()
            x0, y0, x1, y1 = p[0][0], p[0][1], p[1][0], p[1][1]
            if x0 == x1:
                # 往上扩

                is_can_down = True
                if x0 - 1 >= 0:
                    n_p = ((x0 - 1, y0), (x1 - 1, y1))

                    if n_p not in save_st:
                        ce = self.cha_bian_2x2__(x0 - 1, y0, mat)
                        flag = False
                        all_same = True
                        for v_ in ce:
                            if v_ == p:
                                flag = True
                            if v_ not in save_st:
                                all_same = False

                        if all_same:
                            flag = False

                        if d_mat[n_p[0]] < sho:
                            flag = False

                        if flag:
                            is_can_down = False
                            ce = list(ce)
                            if ce[0] == p:
                                save_st.add(ce[1])
                                q.put(ce[1])
                            else:
                                save_st.add(ce[0])
                                q.put(ce[0])
                            x_end = max(x_end, n_p[0][0])
                            y_end = max(y_end, n_p[1][1])
                            mark_st.add((x0 - 1, y0))

                if is_can_down == True:
                    # 往下扩
                    if x0 + 1 < h:
                        n_p = ((x0 + 1, y0), (x1 + 1, y1))
                        if n_p not in save_st:
                            ce = self.cha_bian_2x2__(x0, y0, mat)
                            flag = False
                            all_same = True
                            for v_ in ce:
                                if v_ == p:
                                    flag = True
                                if v_ not in save_st:
                                    all_same = False

                            if all_same:
                                flag = False

                            if d_mat[p[0]] < sho:
                                flag = False

                            if flag:
                                ce = list(ce)
                                if ce[0] == p:
                                    save_st.add(ce[1])
                                    q.put(ce[1])
                                else:
                                    save_st.add(ce[0])
                                    q.put(ce[0])
                                x_end = max(x_end, n_p[0][0])
                                y_end = max(y_end, n_p[1][1])
                                mark_st.add((x0, y0))


            elif y0 == y1:
                # 往左扩
                is_can_right = True
                if y0 - 1 >= 0:
                    n_p = ((x0, y0 - 1), (x1, y1 - 1))
                    if n_p not in save_st:
                        ce = self.cha_bian_2x2__(x0, y0 - 1, mat)
                        flag = False
                        all_same = True
                        for v_ in ce:
                            if v_ == p:
                                flag = True
                            if v_ not in save_st:
                                all_same = False

                        if all_same:
                            flag = False

                        if d_mat[n_p[0]] < sho:
                            flag = False

                        if flag:
                            is_can_right = False
                            ce = list(ce)
                            if ce[0] == p:
                                save_st.add(ce[1])
                                q.put(ce[1])
                            else:
                                save_st.add(ce[0])
                                q.put(ce[0])

                            x_end = max(x_end, n_p[0][1])
                            y_end = max(y_end, n_p[1][1])
                            mark_st.add((x0, y0 - 1))

                if is_can_right == True:
                    # 往右扩
                    if y0 + 1 < w:
                        n_p = ((x0, y0 + 1), (x1, y1 + 1))
                        if n_p not in save_st:
                            ce = self.cha_bian_2x2__(x0, y0, mat)
                            flag = False
                            all_same = True
                            for v_ in ce:
                                if v_ == p:
                                    flag = True
                                if v_ not in save_st:
                                    all_same = False

                            if all_same:
                                flag = False

                            if d_mat[p[0]] < sho:
                                flag = False

                            if flag:
                                ce = list(ce)
                                if ce[0] == p:
                                    save_st.add(ce[1])
                                    q.put(ce[1])
                                else:
                                    save_st.add(ce[0])
                                    q.put(ce[0])
                                x_end = max(x_end, n_p[1][0])
                                y_end = max(y_end, n_p[1][1])
                                mark_st.add((x0, y0))
        return save_st, (x_end, y_end)

    '''
    6-2根据追踪的结果显示边
    '''

    def show_edge_by_mat3_6_2(self, mat, save_st):
        h, w = mat.shape
        bin = np.zeros((h * 2 - 1, w * 2 - 1), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                bin[i * 2, j * 2] = mat[i, j]

        for p in save_st:
            x0, y0, x1, y1 = p[0][0], p[0][1], p[1][0], p[1][1]
            if x0 == x1:
                bin[x0 * 2, y0 * 2 + 1] = 1
            elif y0 == y1:
                bin[x0 * 2 + 1, y0 * 2] = 1

        return bin

    def show_edge_by_mat2_6_2(self, mat, save_st, small_mark_st, big_mark_st):
        h, w = mat.shape
        bin_small = np.zeros((h, w), dtype=np.uint8)
        bin_big = np.zeros((h, w), dtype=np.uint8)

        for p in save_st:
            x0, y0, x1, y1 = p[0][0], p[0][1], p[1][0], p[1][1]

            if mat[x0, y0] < mat[x1, y1]:
                # if len(small_mark_st) != 0 and ((x0, y0)  in small_mark_st):
                # # 先把小边点存进mark_st，如果大边点存在mark_st,则不标记
                #     continue
                small_mark_st.add((x0, y0))

                # if len(big_mark_st) != 0 and ((x1, y1) in big_mark_st):
                #     continue
                big_mark_st.add((x1, y1))

                if (x1, y1) not in small_mark_st and (x0, y0) not in big_mark_st:
                    bin_small[x0, y0] = 1
                    bin_big[x1, y1] = 10
            else:
                small_mark_st.add((x1, y1))
                big_mark_st.add((x0, y0))

                if (x0, y0) not in small_mark_st and (x1, y1) not in big_mark_st:
                    bin_small[x1, y1] = 1
                    bin_big[x0, y0] = 10

        return bin_small, bin_big

    def link_edge_all_mat_by_chabian_v1(self, mat, sho):
        h, w = mat.shape
        x, y = 0, 0
        bin_b_s = np.zeros((h, w), dtype=np.uint8)

        d_mat = self.from_originmat_to_dmat(mat)
        mark_st = set()
        cnt = 0
        while True:

            cnt += 1
            # debug
            print('x,y\t', x, '\t', y)
            x, y = self.find_local_max__(x, y, mat, d_mat, sho)
            if x == -1 and y == -1:
                break

            save_st, end_p = self.start_with_point(x, y, mat, d_mat, 0, mark_st)
            print('save_st', save_st)
            print('end_p', end_p)

            if save_st == []:
                # x, y = end_p[0], end_p[1]
                y += 1
                continue

            if len(save_st) < 4:
                # x, y = end_p[0], end_p[1]
                y += 1
                continue
            m_st, b_st = set(), set()
            for p_ in save_st:
                if mat[p_[0]] < mat[p_[1]]:
                    m_st.add(p_[0])
                    b_st.add(p_[1])
                else:
                    m_st.add(p_[1])
                    b_st.add(p_[0])

            if len(m_st) * 2 < len(b_st) or len(b_st) * 2 < len(m_st):
                y += 1
                continue

            sm, sb = self.show_edge_by_mat2_6_2(mat, save_st, )
            bin_b_s += sm
            bin_b_s += sb
            # x, y = end_p[0], end_p[1] + 1
            y += 1

        # debug
        # print('bin_b_s\n', bin_b_s)
        print('cnt', cnt)
        return bin_b_s

    def link_edge_all_mat_by_chabian(self, mat, sho):
        h, w = mat.shape
        x, y = 0, 0
        bin_b_s = np.zeros((h, w), dtype=np.uint8)

        d_mat = self.from_originmat_to_dmat(mat)

        small_mark_st = set()
        big_mark_st = set()
        cnt = 0
        while True:

            cnt += 1
            # debug
            # print('x,y\t', x, '\t', y)
            x, y = self.find_local_max__(x, y, mat, d_mat, sho)
            if x == -1 and y == -1:
                break

            save_st, end_p = self.start_with_point(x, y, mat, d_mat, 0)
            # print('save_st', save_st)
            # print('end_p', end_p)

            # 优化操作
            if save_st == []:
                # x, y = end_p[0], end_p[1]
                y += 1
                continue
            # 优化操作
            if len(save_st) < 4:
                # x, y = end_p[0], end_p[1]
                y += 1
                continue
            m_st, b_st = set(), set()
            for p_ in save_st:
                if mat[p_[0]] < mat[p_[1]]:
                    m_st.add(p_[0])
                    b_st.add(p_[1])
                else:
                    m_st.add(p_[1])
                    b_st.add(p_[0])
            # 优化操作
            if len(m_st) * 1.5 < len(b_st) or len(b_st) * 1.5 < len(m_st):
                y += 1
                continue

            sm, sb = self.show_edge_by_mat2_6_2(mat, save_st, small_mark_st, big_mark_st)
            bin_b_s += sm
            bin_b_s += sb
            # x, y = end_p[0], end_p[1] + 1
            y += 1

        # debug
        # print('bin_b_s\n', bin_b_s)
        print('cnt', cnt)
        return bin_b_s

        # print('gray\n', gray)
        # print('d_mat\n', d_mat)

        # print(x,'\t', y, '\t', gray[x, y])
        # print('chabian\n',self.cha_bian_2x2__(x, y, gray))
        # print('save_st', save_st)
        # print('bin\n', self.show_edge_by_mat3_6_2(gray, save_st))
        # print('bin_small\n', sm)
        # print('bin_big\n', sb)

    '''
    判断3x3矩阵的中心点是不是局部 前num大
    '''

    def is_local_max_3x3__(self, mat3, num=1):
        # mat3 = d_mat[i - 1: i + 2, j - 1: j + 2]
        mat3 = mat3.flatten()
        cen_val = mat3[4]
        index = np.argsort(mat3)
        # isT = False
        # for k in range(-num, 0):
        #     if mat3[index[k]] == cen_val:  # 4
        #         isT = True
        #         break
        if mat3[index[-1]] == cen_val and mat3[index[-1]] > mat3[index[-2]]:
            return True

        return False

    '''
    判断3x3矩阵的中心点是不是局部 前num小
    '''

    def is_local_min_3x3__(self, mat3, num=1):
        # mat3 = d_mat[i - 1: i + 2, j - 1: j + 2]
        mat3 = mat3.flatten()
        cen_val = mat3[4]
        index = np.argsort(mat3)
        # isT = False
        # for k in range(num):
        #     if mat3[index[k]] == cen_val:  # 4
        #         isT = True
        #         break
        #
        # return isT

        if mat3[index[0]] == cen_val and mat3[index[0]] < mat3[index[1]]:
            return True

        return False

    '''
    如果local_max = True, 表明为整副mat中局部极大值点进行染色，染的颜色为color
    返回染色后的mat
    '''

    def process_local_mat(self, mat, color, local_max=True):
        h, w = mat.shape
        ret_mat = np.zeros(mat.shape, dtype=np.uint8)

        if local_max:
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    mat3 = mat[i - 1: i + 2, j - 1: j + 2]
                    # print(mat3)
                    if self.is_local_max_3x3__(mat3):
                        ret_mat[i, j] = color
        else:
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    mat3 = mat[i - 1: i + 2, j - 1: j + 2]
                    # print(mat3)
                    if self.is_local_min_3x3__(mat3):
                        ret_mat[i, j] = color

        return ret_mat

    def main(self):
        # gray = np.array([
        #     [144, 140, 135, 137, 138, 140, 139],
        #     [113, 150, 142, 139, 141, 138, 138],
        #     [32, 88, 145, 152, 139, 137, 138],
        #     [22, 21, 63, 132, 156, 139, 136],
        #     [30, 27, 18, 44, 110, 150, 148],
        #     [27, 30, 24, 22, 23, 63, 136],
        #     [17, 21, 27, 27, 25, 28, 42],
        #     [22, 26, 28, 29, 31, 27, 27],
        #     [25, 29, 26, 27, 30, 27, 26],
        #     [24, 25, 25, 25, 26, 27, 29],
        #     [23, 21, 24, 27, 26, 25, 24],
        #     [24, 23, 23, 24, 24, 26, 23]
        # ])
        #
        # gray = np.array([
        #     [31,31,31],
        #     [33,33,30],
        #     [24,25,38],
        #     [104,123,147],
        #     [161,155,152]
        # ])

        # gray = np.array([
        #     [10,10],
        #     [10,30]
        # ])
        gray = np.array([
            [125, 120, 122],
            [127, 118, 115],
            [111, 87, 78]
        ])
        # print('fb_3\n', cm.fenbian__(gray))
        # print(self.whether_merge(gray))
        # self.max_difference_mat2_5_13(gray)

        gray = np.array([
            [117, 118, 119, 119, 119, 119, 119],
            [119, 118, 119, 121, 119, 120, 122],
            [120, 124, 113, 114, 120, 120, 120],
            [119, 124, 110, 118, 125, 120, 122],
            [120, 123, 105, 116, 127, 118, 115],
            [123, 120, 101, 109, 111, 87, 78],
            [98, 93, 81, 91, 100, 92, 92],
            [92, 94, 92, 97, 96, 92, 95],
            [96, 98, 94, 92, 90, 91, 89],
        ])
        #
        # gray = np.array([
        #     [29,32,30,31,32,27,24],
        #     [28,26,33,35,31,31,31],
        #     [26,31,27,28,33,33,30],
        #     [33,25,22,21,24,25,38],
        #     [139,120,106,96,104,123,147],
        #     [155,156,153,157,161,155,152],
        #     [143,143,147,144,140,146,148],
        #     [147,145,143,143,144,144,146],
        #     [144,143,144,145,145,143,144]
        # ])
        gray = np.array([
            [104, 100, 95, 88, 88],
            [103, 98, 93, 89, 89],
            [102, 99, 90, 88, 87],
            [101, 100, 88, 89, 88],
            [100, 96, 88, 89, 88]
        ])
        d_mat = dd.from_originmat_to_dmat(gray)
        # bin = self.link_edge_all_mat_by_chabian(gray)

        # self.dif_degree_mat_test(gray)
        # self.d_d_e_a_test(gray)

        # 6-1 test

        # self.eli_useless_edge_allmat_test(gray)

        # test
        # d_mat = dd.from_originmat_to_dmat(gray)
        print('gray\n', gray)
        print('d_mat\n', d_mat)
        # x, y = self.find_local_max__(0, 0, gray, d_mat, 10)
        # print(x,'\t', y, '\t', gray[x, y])
        # print('chabian\n',self.cha_bian_2x2__(x, y, gray))
        save_st, _ = self.start_with_point(1, 2, gray, d_mat)
        # print('save_st', save_st)
        for (x, y) in save_st:
            print("{} = {}\t{}={}".format(x, gray[x], y, gray[y]))
        # print('bin\n', self.show_edge_by_mat3_6_2(gray, save_st))
        # sm, sb = self.show_edge_by_mat2_6_2(gray, save_st)
        # print('bin_small\n', sm)
        # print('bin_big\n', sb)

        # x, y = 0, 0
        # while True:
        #     x, y = self.find_local_max__(x, y, gray, d_mat)
        #     if (x==-1 and y==-1):
        #         break
        #     print(x,'\t', y, '\t', gray[x, y])
        #
        #     y += 1

        # h, w = gray.shape
        # for i in range(h - 1):
        #     for j in range(w - 1):
        #         mat2 = gray[i : i + 2, j : j + 2]
        #         fb3 = cm.fenbian__(mat2)
        #         print('mat2\n', mat2, '\nmat3\n', fb3)


# dd = dis_degree()
# dd.main()
