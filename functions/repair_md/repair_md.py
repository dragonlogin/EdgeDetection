import cv2
import numpy as np
import xlwt


def get22(im, i, j):
    a = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
    return a


def get22Value(im, a):
    b = []
    for i in range(3):
        for j in range(i + 1, 4):
            b.append(abs(int(im[a[i]]) - int(im[a[j]])))
    return b


def findX(b):
    ls = []
    for m, n in b:
        if (m - 1, n - 1) in b:
            ls.append((m, n))
            ls.append((m - 1, n - 1))
            break
        if (m + 1, n + 1) in b:
            ls.append((m, n))
            ls.append((m + 1, n + 1))
            break
        if (m - 1, n + 1) in b:
            ls.append((m, n))
            ls.append((m - 1, n + 1))
            break
        if (m + 1, n - 1) in b:
            ls.append((m, n))
            ls.append((m + 1, n - 1))
            break
    return ls


def getDistance(im, sepa, sepb):
    a = [im[m, n] for m, n in sepa]
    b = [im[m, n] for m, n in sepb]
    value = 0
    amax = max(a)
    amin = min(a)
    bmax = max(b)
    bmin = min(b)
    if amax < bmin:
        value = abs(int(bmin) - int(amax))
    elif bmax < amin:
        value = abs(int(amin) - int(bmax))
    else:
        value = 0
    return value


def getSep22(im):
    sum_c = 0
    fgd = [[[] for i in range(im.shape[1] - 1)] for j in range(im.shape[0] - 1)]  # 存大小边，前面大边后面小边
    sep = [[[] for i in range(im.shape[1] - 1)] for j in range(im.shape[0] - 1)]
    sep_BS = [[[] for i in range(im.shape[1] - 1)] for j in range(im.shape[0] - 1)]
    for i in range(im.shape[0] - 1):
        for j in range(im.shape[1] - 1):
            a = get22(im, i, j)
            baddr = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            sepa = []
            sepb = []
            b = get22Value(im, a)
            if i == 152 and j == 196:
                print(b)
            bv = min(b)
            co = 0
            for x in range(6):
                if bv == b[x]:
                    co += 1
            # print(co)
            if co == 1 or co >= 3:
                for x in range(6):
                    if bv == b[x]:
                        if a[baddr[x][0]] not in sepb:
                            sepb.append(a[baddr[x][0]])
                        if a[baddr[x][1]] not in sepb:
                            sepb.append(a[baddr[x][1]])

            elif co == 2:
                k = [0, 0, 0, 0]
                ls = []
                for x in range(6):
                    if bv == b[x]:
                        ls.append(baddr[x])
                for x in ls:
                    k[x[0]] = 1
                    k[x[1]] = 1
                if sum(k) == 4:
                    c = 0
                    for x in range(6):
                        if bv == b[x]:
                            if c == 0:
                                if a[baddr[x][0]] not in sepb:
                                    sepb.append(a[baddr[x][0]])
                                if a[baddr[x][1]] not in sepb:
                                    sepb.append(a[baddr[x][1]])
                            if c == 1:
                                if a[baddr[x][0]] not in sepa:
                                    sepa.append(a[baddr[x][0]])
                                if a[baddr[x][1]] not in sepa:
                                    sepa.append(a[baddr[x][1]])
                            c += 1
                    # print(sepb,sepa)
                else:
                    for h in range(4):
                        if k[h] == 1:
                            sepb.append(a[h])
                        if k[h] == 0:
                            sepa.append(a[h])
            for x in range(6):
                if bv == b[x]:
                    b[x] = 256
            if len(sepa) + len(sepb) < 4:
                if len(sepb) > 2:
                    for x in a:
                        if x not in sepb:
                            sepa.append(x)
                else:
                    bv = min(b)  # 问题1 第二小的边可能也有好几条，这里选取第一条
                    count = 0  # 改造成功，当第二小边有多条的时候则表示四个点一定是同一区域
                    for x in range(6):
                        if bv == b[x]:
                            count += 1
                    bvIndex = b.index(bv)
                    if a[baddr[bvIndex][0]] in sepb or a[baddr[bvIndex][1]] in sepb:
                        if a[baddr[bvIndex][0]] not in sepb:
                            sepb.append(a[baddr[bvIndex][0]])
                        if a[baddr[bvIndex][1]] not in sepb:
                            sepb.append(a[baddr[bvIndex][1]])
                        for x in a:
                            if x not in sepb:
                                sepa.append(x)
                    else:
                        sepa.append(a[baddr[bvIndex][0]])
                        sepa.append(a[baddr[bvIndex][1]])
                    # 下面注释考虑第二小的有多条的情况
                    # if count==1:
                    #     bvIndex=b.index(bv)
                    #     if a[baddr[bvIndex][0]] in sepb or a[baddr[bvIndex][1]] in sepb:
                    #         if a[baddr[bvIndex][0]] not in sepb:
                    #             sepb.append(a[baddr[bvIndex][0]])
                    #         if a[baddr[bvIndex][1]] not in sepb:
                    #             sepb.append(a[baddr[bvIndex][1]])
                    #         for x in a:
                    #             if x not in sepb:
                    #                 sepa.append(x)
                    #     else:
                    #         sepa.append(a[baddr[bvIndex][0]])
                    #         sepa.append(a[baddr[bvIndex][1]])
                    # elif count==2:
                    #     k = [0,0,0,0]
                    #     ls = []
                    #     for x in range(6):
                    #         if bv == b[x]:
                    #             ls.append(baddr[x])
                    #     for x in ls:
                    #         k[x[0]]=1
                    #         k[x[1]]=1
                    #     if sum(k)==4:
                    #         for x in range(4):
                    #             if k[x]==1 and a[x] not in sepb:
                    #                 sepb.append(a[x])
                    #     else:
                    #         for x in range(4):
                    #             if k[x]==0 and a[x] not in sepb:
                    #                 sepa.append(a[x])
                    #             if k[x]==1 and a[x] not in sepb:
                    #                 sepb.append(a[x])
                    #
                    # else:
                    #     for x in a:
                    #         if x not in sepb:
                    #             sepb.append(x)
            # 增加一步，如果分成两个区域，但是这两个区域的像素值差值小于5则将两个区域合并
            if len(sepa) != 0:
                value = getDistance(im, sepa, sepb)

                if value <= 5:
                    sepb += sepa
                    sepa = []
            # 这里还得加一步判断，当
            sep[i][j].append(sepa)
            sep[i][j].append(sepb)
            if len(sepa) + len(sepb) != 4:
                print(sep[i][j])
            if len(sepa) != 0:
                avalue = [im[x] for x in sepa]
                bvalue = [im[x] for x in sepb]
                amax = max(avalue)
                amin = min(avalue)
                bmax = max(bvalue)
                bmin = min(bvalue)

                blockAV = abs(int(amax) - int(amin))
                blockBV = abs(int(bmax) - int(bmin))

                if amin >= bmax:  # b为小边
                    sep_BS[i][j].append(sepa)
                    sep_BS[i][j].append(sepb)
                    # print(avalue, bvalue, " blockAV ", blockAV, "blockBV", blockBV, "qufen", abs(int(bmax) - int(amin)))
                    # if blockAV+blockBV > abs(int(bmax) - int(amin)):
                    #     # print(i, j)
                    #     sum_c+=1
                    #     sep_BS[i][j]=[]
                    if len(sepa) == 2:
                        fgd[i][j].append(sepa)
                        fgd[i][j].append(sepb)

                    else:
                        fgd[i][j].append([])
                        ls = findX(sepb)
                        fgd[i][j].append(ls)
                if amax <= bmin:  # b为大边
                    sep_BS[i][j].append(sepb)
                    sep_BS[i][j].append(sepa)
                    # print(avalue,bvalue," blockAV ",blockAV,"blockBV",blockBV,"qufen",abs(int(bmax) - int(amin)))
                    # if blockAV+blockBV > abs(int(bmax) - int(amin)):
                    #
                    #     # print(i,j)
                    #     sum_c += 1
                    #     sep_BS[i][j]=[]
                    if len(sepa) == 2:
                        fgd[i][j].append(sepb)
                        fgd[i][j].append(sepa)
                        # print(sepa,sepb)
                    else:
                        ls = findX(sepb)
                        fgd[i][j].append(ls)
                        fgd[i][j].append([])
                # else:
                #     fgd[i][j].append([])
                #     fgd[i][j].append([])
            else:
                fgd[i][j].append([])
                fgd[i][j].append([])
            # print(fgd[i][j])
            # if len(fgd[i][j])==0:
            # print(i,j)
    print(sum_c)
    return fgd, sep, sep_BS


def getQufenAB(im, a, b):
    mina = min(a)
    maxa = max(a)
    minb = min(b)
    maxb = max(b)
    value = 0
    if maxa < minb:
        value = minb - maxa
    elif maxb < mina:
        value = mina - maxb
    else:
        value = 0
    return value


def getEvery2Qufen(im, fgd, sep):
    qufen = np.zeros((len(fgd), len(fgd[0])))
    for i in range(qufen.shape[0]):
        for j in range(qufen.shape[1]):
            if len(sep[i][j][0]) != 0:
                lsa = []
                lsb = []
                for m, n in sep[i][j][0]:
                    lsa.append(int(im[m, n]))
                for m, n in sep[i][j][1]:
                    lsb.append(int(im[m, n]))
                qufen[i, j] = getQufenAB(im, lsa, lsb)
    return qufen


def getNext(im, k, l):
    a = [(k - 1, l - 1), (k - 1, l), (k - 1, l + 1), (k, l + 1), (k + 1, l + 1), (k + 1, l), (k + 1, l - 1),
         (k, l - 1)]
    if k > 0 and k < im.shape[0] - 1:
        if l > 0 and l < im.shape[1] - 1:
            return a
        else:
            b = []
            for m, n in a:
                if m < 0 or m > im.shape[0] - 1 or n < 0 or n > im.shape[1] - 1:
                    pass
                else:
                    b.append((m, n))
            return b
    else:
        b = []
        for m, n in a:
            if m < 0 or m > im.shape[0] - 1 or n < 0 or n > im.shape[1] - 1:
                pass
            else:
                b.append((m, n))
        return b


def getMDP(im, sep_BS):
    gim = im.copy()
    flag = np.zeros(im.shape)
    bsflag = np.zeros(im.shape)  # 0内 1小 2大 3矛盾
    bsafterflag = np.zeros(im.shape)
    for i in range(im.shape[0] - 1):
        for j in range(im.shape[1] - 1):
            p = get22(im, i - 1, j - 1)
            for m, n in p:
                if len(sep_BS[m][n]) > 0:
                    if (i, j) in sep_BS[m][n][0]:
                        flag[i, j] += 10
                    if (i, j) in sep_BS[m][n][1]:
                        flag[i, j] += 1
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # print(flag[i,j])
            if flag[i, j] % 10 > 0 and flag[i, j] // 10 > 0:
                bsflag[i, j] = 3
            if flag[i, j] // 10 > 0 and flag[i, j] % 10 == 0:
                bsflag[i, j] = 2
            if flag[i, j] // 10 == 0 and flag[i, j] % 10 > 0:
                bsflag[i, j] = 1
    for i in range(1, im.shape[0]):
        for j in range(1, im.shape[1]):
            if bsflag[i, j] == 1:
                bsafterflag[i, j] = 1
            if bsflag[i, j] == 2:
                bsafterflag[i, j] = 2

            if bsflag[i, j] == 3:
                addr = getNext(im, i, j)
                ad = []
                for x, y in addr:
                    if bsflag[x, y] != 3:
                        ad.append((x, y))
                adv = [im[x, y] for x, y in ad]
                # print(ad,addr)
                adx = [abs(int(im[i, j]) - int(x)) for x in adv]
                if len(adx) == 0:
                    continue
                bsafterflag[i, j] = bsflag[ad[adx.index(min(adx))]]
                gim[i, j] = im[ad[adx.index(min(adx))]]
            # if bsflag[i,j]==3:
            #     addr = getNext(im,i,j)
            #     addrV = [im[x,y] for x,y in addr]
            #     a,b = sep822(addr, addrV)
            #     aV = [im[x,y] for x,y in a]
            #     bV = [im[x,y] for x,y in b]
            #     meana = np.mean(aV)
            #     meanb = np.mean(bV)
            #     # if i==12 and j==12:
            #     #     print(aV, bV,a,b,meana,meanb)
            #     if(abs(int(meana)-int(im[i,j]))>abs(int(meanb)-int(im[i,j]))):
            #         if meana>meanb:
            #             bsafterflag[i,j] = 1
            #         else:
            #             bsafterflag[i,j] = 2
            #     else:
            #         if meana>meanb:
            #             bsafterflag[i,j] = 2
            #         else:
            #             bsafterflag[i,j] = 1
    return bsflag, bsafterflag, gim


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_linkpoint2(point_copy, g1, g2, g3, savefile):
    pattern = xlwt.Pattern()
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = 7
    style = xlwt.XFStyle()
    style.pattern = pattern

    pattern1 = xlwt.Pattern()
    pattern1.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern1.pattern_fore_colour = 5
    style1 = xlwt.XFStyle()
    style1.pattern = pattern1

    pattern2 = xlwt.Pattern()
    pattern2.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern2.pattern_fore_colour = 3
    style2 = xlwt.XFStyle()
    style2.pattern = pattern2

    pattern3 = xlwt.Pattern()
    pattern3.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern3.pattern_fore_colour = 16
    style3 = xlwt.XFStyle()
    style3.pattern = pattern3

    pattern4 = xlwt.Pattern()
    pattern4.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern4.pattern_fore_colour = 31
    style4 = xlwt.XFStyle()
    style4.pattern = pattern4

    pattern5 = xlwt.Pattern()
    pattern5.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern5.pattern_fore_colour = 40
    style5 = xlwt.XFStyle()
    style5.pattern = pattern5

    pattern6 = xlwt.Pattern()
    pattern6.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern6.pattern_fore_colour = 50
    style6 = xlwt.XFStyle()
    style6.pattern = pattern6

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('m', cell_overwrite_ok=True)
    tj = 0
    for j in range(point_copy.shape[1] - 1):
        if tj > 250:
            tj = 0
            name_sheet = 'm' + str(int(j / 255))
            sheet = book.add_sheet(name_sheet, cell_overwrite_ok=True)
        for i in range(point_copy.shape[0] - 1):
            sss = int(point_copy[i][j])
            # if (count_linkpoint[i][j] == 0):
            #     # pass
            #     sheet.write(i, tj, sss)  # 天蓝色
            if g2[i, j] == 1:
                sheet.write(i, tj, str(sss) + "/" + str(g1[i, j]), style1)  # 黄色
            elif g2[i, j] == 2:
                sheet.write(i, tj, str(sss) + "/" + str(g1[i, j]), style2)  # 绿色
            elif g2[i, j] == 3 and g3[i, j] == 1:
                sheet.write(i, tj, str(sss) + "/" + str(g1[i, j]), style3)
            elif g2[i, j] == 3 and g3[i, j] == 2:
                sheet.write(i, tj, str(sss) + "/" + str(g1[i, j]), style4)
            else:
                sheet.write(i, tj, str(sss) + "/" + str(g1[i, j]))
        tj = tj + 1
    book.save(savefile)


def fixMD(im):
    fgd, sep, sep_BS = getSep22(im)
    qufen = getEvery2Qufen(im, fgd, sep)
    bsflag, bsafterflag, gim = getMDP(im, sep_BS)
    return gim


# imagePath = 'BSR/bench/data/images/8068.jpg'
# im = cv2.imread(imagePath, 0)
# fgd, sep, sep_BS = getSep22(im)
# qufen = getEvery2Qufen(im, fgd, sep)
# bsflag, bsafterflag, gim = getMDP(im, sep_BS)
# cv_show("gim", gim)
# write_linkpoint2(gim, qufen, bsflag, bsafterflag, "gim.xls")

class repair_md(object):
    def fix_md(self, mat):
        return fixMD(mat)

    def fix_md_test(self):
        mat = np.array([
            [10, 10, 22],
            [10, 20, 22],
            [60, 60, 60]
        ])
        ret_mat = self.fix_md(mat)
        print('ret_mat\n', ret_mat)

# rmd = repair_md()
# rmd.fix_md_test()
