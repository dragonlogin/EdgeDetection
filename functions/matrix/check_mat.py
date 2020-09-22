import numpy as np
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from functions.matrix.check_num import check_num

# from func.mergeRegion import MergeRegion
cn = check_num()

'''
class功能：只对外调用bsmall_v2(gray)->markMat,show_edge(martMat)->bin_img
传入一个原始矩阵，经过连边，大小边判断，矛盾点判断，可能的边缘信息判断，返回一个边缘点矩阵
'''


class check_mat(object):

    def bigedge_smalledge_index(self, fb_mat3):
        lx, ly = 0, 0
        p2 = [(lx, ly), (lx, ly + 1), (lx + 1, ly), (lx + 1, ly + 1)]
        p3 = [(lx, ly), (lx, ly + 2), (lx + 2, ly), (lx + 2, ly + 2)]
        # print('p3', p3)
        sei, bei = [], []
        mark_mat = []

        if self.check4__(lx, ly, fb_mat3):
            # print('444444444')
            return sei, bei
        elif self.check3__(lx, ly, fb_mat3):
            # print('333333333333')
            mark_mat = self.mark_bsm3__(lx, ly, fb_mat3)
        elif self.check2__(lx, ly, fb_mat3):
            # print('222222222222')
            mark_mat = self.mark_bsm2__(lx, ly, fb_mat3)
        # print('\nmark_mat\n', mark_mat)

        for i in range(len(p3)):
            if mark_mat[p3[i]] == 5:
                sei.append(p2[i])
            elif mark_mat[p3[i]] == 6:
                bei.append(p2[i])
        # print('sei', sei)
        # print('bei', bei)

        return sei, bei

    # test
    def bigedge_smalledge_index_test(self, mat2):
        fb_mat3 = self.fenbian__(mat2)
        # print('\n', fb_mat3)

        sei, bei = self.bigedge_smalledge_index(fb_mat3)
        se_val = [mat2[i] for i in sei]
        be_val = [mat2[i] for i in bei]

        # print('se\n', se_val)
        # print('be\n', be_val)

    # 检查一个3x3连边矩阵，是否有4条边相连
    # 返回True False
    def check4__(self, x, y, new_gray) -> bool:

        edge_l = [(x, y + 1), (x + 1, y + 2), (x + 2, y + 1), (x + 1, y), (x + 1, y + 1)]
        pix_l = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]

        if new_gray[edge_l[4]] == 0:
            # if new_gray[edge_l[0]]>0 and new_gray[edge_l[2]]>0 and new_gray[edge_l[1]]==0 and new_gray[edge_l[3]]==0:
            #     return False
            # if new_gray[edge_l[1]]>0 and new_gray[edge_l[3]]>0 and new_gray[edge_l[0]]==0 and new_gray[edge_l[2]]==0:
            #     return False
            el = []
            for i in range(4):
                el.append(new_gray[edge_l[i]])
            el.sort(reverse=True)

            if el[0] == 0:
                return True
            if el[2] == 0:
                return False
            else:
                return True

        else:
            subl = []
            for i in range(4):
                subl.append(new_gray[edge_l[i]])
            subl.sort(reverse=True)
            if (new_gray[edge_l[4]] == 3 or new_gray[edge_l[4]] == 4) and (subl[1] == 0):
                return False

            else:
                if new_gray[edge_l[4]] == 3:
                    # ad
                    if new_gray[edge_l[0]] > 0 and new_gray[edge_l[1]] > 0 and new_gray[edge_l[2]] == 0 and new_gray[
                        edge_l[3]] == 0:
                        return False
                    if new_gray[edge_l[2]] > 0 and new_gray[edge_l[3]] > 0 and new_gray[edge_l[0]] == 0 and new_gray[
                        edge_l[1]] == 0:
                        return False

                elif new_gray[edge_l[4]] == 4:
                    # ad
                    if new_gray[edge_l[0]] > 0 and new_gray[edge_l[3]] > 0 and new_gray[edge_l[1]] == 0 and new_gray[
                        edge_l[2]] == 0:
                        return False
                    if new_gray[edge_l[1]] > 0 and new_gray[edge_l[2]] > 0 and new_gray[edge_l[0]] == 0 and new_gray[
                        edge_l[3]] == 0:
                        return False

        return True

    # 检查3x3连边矩阵，是否有2条边相连
    # 返回true false
    def check2__(self, x, y, new_gray) -> bool:
        if self.check4__(x, y, new_gray):
            return False

        else:
            edge_l = [(x, y + 1), (x + 1, y + 2), (x + 2, y + 1), (x + 1, y), (x + 1, y + 1)]
            pix_l = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]

            if new_gray[edge_l[4]] == 3 or new_gray[edge_l[4]] == 4:
                return False
            else:
                if new_gray[edge_l[0]] > 0 and new_gray[edge_l[2]] > 0 and new_gray[edge_l[1]] == 0 and new_gray[
                    edge_l[3]] == 0:
                    return True

                if new_gray[edge_l[1]] > 0 and new_gray[edge_l[3]] > 0 and new_gray[edge_l[0]] == 0 and new_gray[
                    edge_l[2]] == 0:
                    return True

        return False

    # 检查3x3连边矩阵，是否有3条边相连
    # 返回true false
    def check3__(self, x, y, new_gray) -> bool:
        if self.check2__(x, y, new_gray) or self.check4__(x, y, new_gray):
            return False
        return True

    # 传入一个3x3连边数组，
    # 返回一个3x3标记矩阵，端点为5表示小边，6表示大边，边缘>0表示为可能的边缘
    def mark_bsm2__(self, x, y, new_gray):
        x, y = 0, 0
        mark = np.zeros((3, 3), np.uint8)
        edgel = [(x, y + 1), (x + 1, y + 2), (x + 2, y + 1), (x + 1, y)]
        pixl = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]
        medge1 = [(0, 1), (1, 2), (2, 1), (1, 0)]
        mpixl = [(0, 0), (0, 2), (2, 2), (2, 0)]

        if new_gray[edgel[0]] > 0:
            mark[medge1[0]] += 1
            mark[medge1[2]] += 1

            if max(new_gray[pixl[0]], new_gray[pixl[1]]) < min(new_gray[pixl[2]], new_gray[pixl[3]]):
                mark[mpixl[0]] = mark[mpixl[1]] = 5
                mark[mpixl[2]] = mark[mpixl[3]] = 6
            else:
                mark[mpixl[0]] = mark[mpixl[1]] = 6
                mark[mpixl[2]] = mark[mpixl[3]] = 5

        elif new_gray[edgel[1]] > 0:
            mark[medge1[1]] += 1
            mark[medge1[3]] += 1

            if max(new_gray[pixl[1]], new_gray[pixl[2]]) < min(new_gray[pixl[0]], new_gray[pixl[3]]):
                mark[mpixl[1]] = mark[mpixl[2]] = 5
                mark[mpixl[0]] = mark[mpixl[3]] = 6
            else:
                mark[mpixl[1]] = mark[mpixl[2]] = 6
                mark[mpixl[0]] = mark[mpixl[3]] = 5

        return mark

    # 传入一个3x3 连边mat
    # 返回一个3x3标记矩阵，端点为5表示小边，6表示大边，中心点值为3，表示主对角线为可能的边缘，为4表示副对角线为可能的边缘
    # new_gray 为3x3的连边数组
    def mark_bsm3__(self, x, y, new_gray):
        x, y = 0, 0
        mark = np.zeros((3, 3), np.uint8)
        # 连边mat 的edge， pix
        edgel = [(x, y + 1), (x + 1, y + 2), (x + 2, y + 1), (x + 1, y), (x + 1, y + 1)]
        pixl = [(x, y), (x, y + 2), (x + 2, y + 2), (x + 2, y)]

        # 标记mat的 的edge， pix
        medge1 = [(0, 1), (1, 2), (2, 1), (1, 0), (1, 1)]
        mpixl = [(0, 0), (0, 2), (2, 2), (2, 0)]

        if new_gray[edgel[4]] == 3:
            mark[medge1[4]] = 3

            if new_gray[edgel[0]] > 0 or new_gray[edgel[1]] > 0:
                # abd
                al = [new_gray[pixl[0]], new_gray[pixl[1]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[3]]

                if bval >= max_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[2]] = 5
                    mark[mpixl[3]] = 6
                elif bval <= min_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[2]] = 6
                    mark[mpixl[3]] = 5

            elif new_gray[edgel[2]] > 0 or new_gray[edgel[3]] > 0:
                # acd
                al = [new_gray[pixl[0]], new_gray[pixl[3]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[1]]

                if bval > max_:
                    mark[mpixl[0]] = mark[mpixl[3]] = mark[mpixl[2]] = 5
                    mark[mpixl[1]] = 6
                elif bval < min_:
                    mark[mpixl[0]] = mark[mpixl[3]] = mark[mpixl[2]] = 6
                    mark[mpixl[1]] = 5

        elif new_gray[edgel[4]] == 4:
            mark[medge1[4]] = 4

            if new_gray[edgel[0]] > 0 or new_gray[edgel[3]] > 0:
                # abc
                al = [new_gray[pixl[0]], new_gray[pixl[1]], new_gray[pixl[3]]]

                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[2]]

                if bval > max_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[3]] = 5
                    mark[mpixl[2]] = 6
                elif bval < min_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[3]] = 6
                    mark[mpixl[2]] = 5

            elif new_gray[edgel[1]] > 0 or new_gray[edgel[2]] > 0:
                # bcd
                al = [new_gray[pixl[1]], new_gray[pixl[3]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[0]]

                if bval > max_:
                    mark[mpixl[1]] = mark[mpixl[3]] = mark[mpixl[2]] = 5
                    mark[mpixl[0]] = 6
                elif bval < min_:
                    mark[mpixl[1]] = mark[mpixl[3]] = mark[mpixl[2]] = 6
                    mark[mpixl[0]] = 5

        else:  # center = 0
            if new_gray[edgel[0]] > 0 and new_gray[edgel[1]] > 0:
                # abd
                mark[medge1[4]] = 3
                al = [new_gray[pixl[0]], new_gray[pixl[1]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[3]]

                if bval > max_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[2]] = 5
                    mark[mpixl[3]] = 6
                elif bval < min_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[2]] = 6
                    mark[mpixl[3]] = 5

            elif new_gray[edgel[1]] > 0 and new_gray[edgel[2]] > 0:
                # bcd
                mark[medge1[4]] = 4
                al = [new_gray[pixl[1]], new_gray[pixl[3]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[0]]

                if bval > max_:
                    mark[mpixl[1]] = mark[mpixl[3]] = mark[mpixl[2]] = 5
                    mark[mpixl[0]] = 6
                elif bval < min_:
                    mark[mpixl[1]] = mark[mpixl[3]] = mark[mpixl[2]] = 6
                    mark[mpixl[0]] = 5

            elif new_gray[edgel[2]] > 0 and new_gray[edgel[3]] > 0:
                # acd
                mark[medge1[4]] = 3
                al = [new_gray[pixl[0]], new_gray[pixl[3]], new_gray[pixl[2]]]
                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[1]]

                if bval > max_:
                    mark[mpixl[0]] = mark[mpixl[3]] = mark[mpixl[2]] = 5
                    mark[mpixl[1]] = 6
                elif bval < min_:
                    mark[mpixl[0]] = mark[mpixl[3]] = mark[mpixl[2]] = 6
                    mark[mpixl[1]] = 5

            elif new_gray[edgel[3]] > 0 and new_gray[edgel[0]] > 0:
                # abc
                mark[medge1[4]] = 4
                al = [new_gray[pixl[0]], new_gray[pixl[1]], new_gray[pixl[3]]]

                max_, min_ = max(al), min(al)
                bval = new_gray[pixl[2]]

                if bval > max_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[3]] = 5
                    mark[mpixl[2]] = 6
                elif bval < min_:
                    mark[mpixl[0]] = mark[mpixl[1]] = mark[mpixl[3]] = 6
                    mark[mpixl[2]] = 5

        return mark

    # 传入一个原始矩阵mxn
    # 返回一个2*m-1, 2*n-1的标记矩阵：含有大6小5边，矛盾点7，可能的边缘，水平竖直边缘>0,正对角线3，副4
    def bmsall_v2(self, gray, sho=0):
        h, w = gray.shape
        nh, nw = 2 * h - 1, 2 * w - 1
        mark = np.zeros((nh, nw), dtype=np.uint8)

        # print('new_gray is in function of bmsall_v2 of class of CheckMat')
        for i in range(0, nh - 1, 2):
            for j in range(0, nw - 1, 2):
                ii, jj = i // 2, j // 2
                # print(ii, jj)
                mat = gray[ii:ii + 2, jj:jj + 2]
                ls = [gray[ii, jj], gray[ii, jj + 1], gray[ii + 1, jj], gray[ii + 1, jj + 1]]

                if max(ls) - min(ls) <= sho:
                    continue

                new_gray = self.fenbian__(mat)
                # print('new_gray=\n{}'.format(new_gray))
                if self.check4__(0, 0, new_gray):
                    continue
                else:
                    if self.check2__(0, 0, new_gray):
                        # if mr.merge2(mat) == True:
                        #     continue
                        mark2 = self.mark_bsm2__(i, j, new_gray)
                    elif self.check3__(0, 0, new_gray):
                        # if mr.merge3(mat):
                        #     continue
                        mark2 = self.mark_bsm3__(i, j, new_gray)

                    medge1 = [(0, 1), (1, 2), (2, 1), (1, 0), (1, 1)]
                    mpixl = [(0, 0), (0, 2), (2, 2), (2, 0)]

                    for (x, y) in medge1:
                        mark[x + i, y + j] += mark2[x, y]

                    for (x, y) in mpixl:
                        if mark[x + i, y + j] == 0:
                            mark[x + i, y + j] = mark2[x, y]
                        elif mark[x + i, y + j] == 5 and mark2[x, y] == 6:
                            mark[x + i, y + j] = 7
                        elif mark[x + i, y + j] == 6 and mark2[x, y] == 5:
                            mark[x + i, y + j] = 7

                # print('mark=\n{}'.format(mark))
        return mark

    # 没用
    def bmsall(self, new_gray):
        h, w = new_gray.shape
        mark = np.zeros(new_gray.shape, dtype=np.uint8)

        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                if self.check4(i, j, new_gray):
                    continue
                else:
                    if self.check2(i, j, new_gray):
                        mark2 = self.mark_bsm2(i, j, new_gray)
                    elif self.check3(i, j, new_gray):
                        mark2 = self.mark_bsm3(i, j, new_gray)

                    medge1 = [(0, 1), (1, 2), (2, 1), (1, 0), (1, 1)]
                    mpixl = [(0, 0), (0, 2), (2, 2), (2, 0)]

                    for (x, y) in medge1:
                        mark[x + i, y + j] += mark2[x, y]
                    for (x, y) in mpixl:
                        if mark[x + i, y + j] == 0:
                            mark[x + i, y + j] = mark2[x, y]
                        elif mark[x + i, y + j] == 5 and mark2[x, y] == 6:
                            mark[x + i, y + j] = 1
                        elif mark[x + i, y + j] == 6 and mark2[x, y] == 5:
                            mark[x + i, y + j] = 1

        return mark

    # 传入一个左上角的坐标和2x2原始矩阵
    # 返回 一个3x3连边矩阵
    def fenbian__(self, mat2, x=0, y=0):
        # mark = np.zeros(gray.shape, dtype=np.uint32)
        # h, w = gray.shape
        # mp = {}
        # x, y 左上角坐标
        '''
        A B
        C D
        A : (x, y)
        B : (x1, y1)
        D : (x2, y2)
        C : (x3, y3)
        gray = [
            [55,56],
            [58,63]
        ]
        '''
        h, w = mat2.shape
        nh, nw = h * 2 - 1, w * 2 - 1
        fb_mat3 = np.zeros((nh, nw), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                fb_mat3[i * 2, j * 2] = mat2[i, j]

        # ck = CheckNum()
        def mat_2x2__(x, y, node=1):
            # x, y => 0
            x1, y1 = x, y + 1  # => 1
            x2, y2 = x + 1, y + 1  # =>2
            x3, y3 = x + 1, y  # => 3
            ab = abs(int(mat2[x, y]) - int(mat2[x1, y1]))
            bd = abs(int(mat2[x1, y1]) - int(mat2[x2, y2]))
            dc = abs(int(mat2[x3, y3]) - int(mat2[x2, y2]))
            ac = abs(int(mat2[x, y]) - int(mat2[x3, y3]))
            ad = abs(int(mat2[x, y]) - int(mat2[x2, y2]))
            bc = abs(int(mat2[x1, y1]) - int(mat2[x3, y3]))
            # print(ab,bd,dc,ac,ad,bc)
            edge_list = [ab, bd, dc, ac, ad, bc]
            # print(edge_list)
            sort_index = np.argsort(edge_list).tolist()
            # print('sort_index=\n{}'.format(sort_index))
            cnt = 0  # 最小值的个数
            min_ = edge_list[sort_index[0]]

            for i in sort_index:
                # print(edge_list[i])
                if edge_list[i] == min_:
                    cnt += 1

            def del4or5(cnt):
                index_list = sort_index[: cnt]
                if cn.check_ij(4, 5, index_list) == 2:
                    if index_list.index(4) < index_list.index(5):
                        sort_index.remove(5)
                    else:
                        sort_index.remove(4)

            if cnt > 1:
                del4or5(cnt)
                cn.cpu(fb_mat3, sort_index, cnt)
            elif cnt == 1:
                s_min = edge_list[sort_index[1]]
                s_cnt = 0
                for i in sort_index:
                    if edge_list[i] == s_min:
                        s_cnt += 1

                del4or5(cnt + s_cnt)
                cn.cpu(fb_mat3, sort_index, cnt + s_cnt)

        mat_2x2__(0, 0)
        # print('gray=\n{}\tnew_gray=\n{}'.format(gray, new_gray))
        # print("check={}\t{}\t{}\t\n".format(ct.check2(0,0, new_gray), ct.check3(0,0,new_gray), ct.check4(0,0,new_gray)))
        return fb_mat3

    # test
    def fenbian_test(self, origin_gary):
        h, w = origin_gary.shape

        for i in range(h - 1):
            for j in range(w - 1):
                gray = origin_gary[i:i + 2, j:j + 2]
                self.fenbian__(gray)

    # 传入一个3x3 标记边缘点和矛盾点的矩阵，
    # 返回一个2x2 0 1, 边缘矩阵
    def edge_2x2__(self, new_gray):
        edge = np.zeros((2, 2), dtype=np.uint8)
        edgel = [(0, 1), (1, 2), (2, 1), (1, 0), (1, 1)]
        pixl = [(0, 0), (0, 2), (2, 2), (2, 0)]
        el = [(0, 0), (0, 1), (1, 1), (1, 0)]

        # ab
        if new_gray[pixl[0]] == 7 and new_gray[pixl[1]] == 7 and new_gray[edgel[0]] > 0:
            edge[el[0]] = edge[el[1]] = 1
        if new_gray[pixl[0]] == 7 and new_gray[pixl[2]] == 7 and new_gray[edgel[4]] == 3:
            edge[el[0]] = edge[el[2]] = 1
        if new_gray[pixl[0]] == 7 and new_gray[pixl[3]] == 7 and new_gray[edgel[3]] > 0:
            edge[el[0]] = edge[el[3]] = 1
        if new_gray[pixl[1]] == 7 and new_gray[pixl[3]] == 7 and new_gray[edgel[4]] == 4:
            edge[el[1]] = edge[el[3]] = 1

        return edge

    # new_edge 2*m-1 , 2*n-1
    # 传入一个标记边缘和矛盾点的矩阵2*m-1, 2*n-1
    # 返回一个mXn的0 1 边缘矩阵
    def show_edge(self, new_gray):
        h, w = new_gray.shape
        bin_img = np.zeros(((h + 1) // 2, (w + 1) // 2), dtype=np.uint8)

        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                arr_3x3 = new_gray[i:i + 3, j:j + 3]
                edge = self.edge_2x2__(arr_3x3)
                ii, jj = i // 2, j // 2
                el = [(0, 0), (0, 1), (1, 1), (1, 0)]

                for (x, y) in el:
                    if bin_img[ii + x, jj + y] == 0:
                        bin_img[ii + x, jj + y] = edge[x, y]

        return bin_img

    '''
    @:param 传入一个原始mat m*n
    @:return 返回一个矛盾点mat m*n
    '''

    def from_gray_to_md_mat(self, in_mat):
        mark_mat = self.bmsall_v2(in_mat)
        h, w = in_mat.shape
        md_mat = np.zeros(in_mat.shape, dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                ii, jj = i * 2, j * 2
                if mark_mat[ii, jj] == 7:
                    md_mat[i, j] = 1

        return md_mat


'''
class功能：只对外调用bsmall_v2(gray)->markMat,show_edge(martMat)->bin_img
传入一个原始矩阵，经过连边，大小边判断，矛盾点判断，可能的边缘信息判断，返回一个边缘点矩阵
'''

# ct = CheckMat()
# arr = [
#     [37,46],
#     [45, 50]
# ]
# arr = np.array(arr)
# ret = ct.fenbian(0,0, arr)
# print(ret)
# gray = [
#     [31,1,37],
#     [0,0,1],
#     [122,0,45]
# ]

# gray = np.array(gray)
#
# gray = np.array([
#     [30,0,20],
#     [0,3,1],
#     [10,1,50]
# ])
# gray = [
#     [118,115],
#     [113,114]
# ]
# gray = [
#     [136, 122, 157, 166, 177],
#     [126, 118, 148, 155, 166],
#     [118, 115, 139, 147, 159],
#     [113, 114, 130, 138, 153],
#     [104, 117, 133, 142, 159],
# ]
# ct.fenbian_test(np.array(gray))
# gray = np.array(gray)
# for i in range(2):
#     for j in range(2):
#         print(gray[i:i+2,j:j+2])
# ret4 = ct.check4(0,0, gray)
# ret3 = ct.check3(0, 0, gray)
# ret2 = ct.check2(0,0, gray)
# print(ret2, ret3, ret4)
#
# # mark = ct.mark_bsm2(0, 0, gray)
# # print(mark)
# mark = ct.mark_bsm3(0, 0, gray)
# print(mark)
# mark = ct.bmsall(gray)
# print(mark)
