import numpy as np
# 对 3 * 3 矩阵进行连边操作

class check_num(object):

    # def check45(self, index):
    #     if (3 in index) and (5 in index):
    #         return 1
    #     elif (3 not in index) and (5 not in index):
    #         return -1
    #     return 0
    #
    # def check01(self, index):
    #     if (0 in index) and (1 in index):
    #         return 2
    #     elif (0 not in index) and (1 not in index):
    #         return 0
    #     return 1

    def check_ij(self, i, j, index):
        if (i in index) and (j in index):
            return 2
        elif (i not in index) and (j not in index):
            return 0
        return 1

    def link_edge2(self, new_gray, node_index, nx=0, ny=0):
        # x,y为gray的2x2左上角坐标
        if node_index == 0:
            mx, my = nx, ny + 1
            new_gray[mx, my] += 1
        elif node_index == 1:
            mx, my = nx + 1, ny + 2
            new_gray[mx, my] += 1
        elif node_index == 2:
            mx, my = nx + 2, ny + 1
            new_gray[mx, my] += 1
        elif node_index == 3:
            mx, my = nx + 1, ny
            new_gray[mx, my] += 1
        elif node_index == 4:
            mx, my = nx + 1, ny + 1
            new_gray[mx, my] = 3
        elif node_index == 5:
            mx, my = nx + 1, ny + 1
            new_gray[mx, my] = 4

    def link_list_edge(self, new_gray, list, x=0, y=0):
        for node_index in list:
            self.link_edge2(new_gray, node_index)

    def pro2(self, new_gray, index):
        if self.check_ij(4, 5, index) == 2:
            return False
        elif self.check_ij(4, 5, index) == 0:
            if self.check_ij(0, 1, index) == 2:
                self.link_list_edge(new_gray, [0, 1, 4])
            elif self.check_ij(0, 2, index) == 2:
                self.link_list_edge(new_gray, [0, 2])
            elif self.check_ij(0, 3, index) == 2:
                self.link_list_edge(new_gray, [0, 3, 5])
            elif self.check_ij(1, 2, index) == 2:
                self.link_list_edge(new_gray, [1, 2, 5])
            elif self.check_ij(1, 3, index) == 2:
                self.link_list_edge(new_gray, [1, 3])
            elif self.check_ij(2, 3, index) == 2:
                self.link_list_edge(new_gray, [2, 3, 4])
        elif 4 in index:
            if self.check_ij(0, 4, index) == 2:
                self.link_list_edge(new_gray, [0, 1, 4])
            elif self.check_ij(1, 4, index) == 2:
                self.link_list_edge(new_gray, [0, 1, 4])
            elif self.check_ij(2, 4, index) == 2:
                self.link_list_edge(new_gray, [2, 3, 4])
            elif self.check_ij(3, 4, index) == 2:
                self.link_list_edge(new_gray, [2, 3, 4])
        elif 5 in index:
            if self.check_ij(0, 5, index) == 2:
                self.link_list_edge(new_gray, [0, 3, 5])
            elif self.check_ij(1, 5, index) == 2:
                self.link_list_edge(new_gray, [1, 2, 5])
            elif self.check_ij(2, 5, index) == 2:
                self.link_list_edge(new_gray, [1, 2, 5])
            elif self.check_ij(3, 5, index) == 2:
                self.link_list_edge(new_gray, [0, 3, 5])
        return True

    def pro3(self, new_gray, index):
        if self.check_ij(4, 5, index) == 2 or self.check_ij(4, 5, index) == 0:
            return False
        else:
            if 4 in index:
                if self.check_ij(0, 1, index) == 2:
                    self.link_list_edge(new_gray, [0, 1, 4])
                elif self.check_ij(2, 3, index) == 2:
                    self.link_list_edge(new_gray, [2, 3, 4])
            elif 5 in index:
                if self.check_ij(0, 3, index) == 2:
                    self.link_list_edge(new_gray, [0, 3, 5])
                elif self.check_ij(1, 2, index) == 2:
                    self.link_list_edge(new_gray, [1, 2, 5])
        return True

    def cpu(self, new_gray, index, cnt):
        if cnt >= 4:
            return
        elif cnt == 3:
            self.pro3(new_gray, index[:3])
        elif cnt == 2:
            self.pro2(new_gray, index[:2])

index = np.array([2,3,4])
index = [2,3]
new_gray = np.zeros((3,3), dtype=np.uint8)

ck = check_num()
print(new_gray)
ck.link_list_edge(new_gray, index)
ret = ck.pro2(new_gray, index)
print(new_gray)
# print(ret)
print(ck.pro3(new_gray, [2,3,4]))
print(new_gray)

# print(self.check45(index))
# print(self.check01(index))
# print(self.check_ij(3,4,index))
# print(self.check_ij(2,3,index))
