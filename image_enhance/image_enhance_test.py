from image_enhance.ima_enhance import ima_enhance
ie = ima_enhance()
import numpy as np
class image_enhance_test(object):
    def __init__(self):

        self.gray = np.array([
            [10, 10, 50],
            [20, 70, 20],
            [20, 20, 20]
        ])

        # 测试样例1
        self.gray = np.array([
            [10, 10, 10],
            [30, 50, 30],
            [30, 30, 30]
        ])

        # 测试样例2
        self.gray = np.array([
            [10, 10, 10],
            [30, 20, 30],
            [30, 30, 30]
        ])
        # 测试样例3
        self.gray = np.array([
            [10, 10, 10],
            [30, 15, 30],
            [30, 30, 30]
        ])
        # 测试样例4
        self.gray = np.array([
            [30, 30, 30],
            [30, 15, 30],
            [30, 30, 30]
        ])
        # 测试样例5
        self.gray = np.array([
            [10, 30, 30],
            [30, 15, 30],
            [30, 30, 30]
        ])

        # self.gray = np.array(gray, np.uint8)
        # filePath = './image/lena.jpg'
        # filePath = './image/3.png'
        # filePath = './image/3-1.png'
        # filePath = './image/3-2.png'
        # filePath = './image/3-3.png'
        # filePath = './image/3-5.png'
        # filePath = './image/2.png'
        # filePath = './image/figure.jpg'
        # filePath = './image/t1.jpg'
        # filePath = './image/a.png'


if __name__ == '__main__':
    main = image_enhance_test()
    gray = main.gray
    A, B = ie.get_AB(gray, 1, 1)
    print('gray=\n{}\n'.format(gray))
    print('A={}\nB={}\n'.format(A, B))

    ie.replace_with_median(gray)
    print('gray=\n{}\n'.format(gray))

    # a = [10, 10, 20, 20, 20, 20, 20]
    # val = ie.get_median(a)
    # v2 = ie.get_mean(a)
    # print(val)
    # print(v2)

    pass