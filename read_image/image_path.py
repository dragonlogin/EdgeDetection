# truthPath = '../image/8068.mat'
# imgPath = '../image/8068.jpg'
import cv2


class image_path:
    def getImagePath(self):
        # return '../image/8068.jpg' # 天鹅
        # return '../image/qie.jpg' # 部分企鹅图片

        return '../image/leaf.jpg'
        # return '../image/lena.jpg'

        # return '../image/house.jpg'
        # return '../BSDS500/data/images/train/35008.jpg'
        # return '../image/figure.jpg'
        # return '../BSDS500/data/images/val/3096.jpg' # 飞机
        # return '../BSDS500/data/images/train/24063.jpg' #教堂
        # return '../BSDS500/data/images/train/22013.jpg' # 楼房
        # return '../BSDS500/data/images/train/15004.jpg' # 老人
        # return '../BSDS500/data/images/train/135069.jpg' # 老鹰
        # return '../BSDS500/data/images/train/41004.jpg'  # 牛
        # return '../BSDS500/data/images/test/393035.jpg'  # 论文中船
        # return '../BSDS500/data/images/test/384022.jpg'  # 论文中船

        '''
        人体模特
        '''
        # return '../image/man-picture/1-1.JPG'  #
        # return '../image/man-picture/1-5.JPG'  #

    def getMatPath(self):
        # return '../image/8068.mat'
        # return '../BSDS500/data/groundTruth/train/35008.mat'
        # return '../BSDS500/data/groundTruth/val/3096.mat'
        # return '../BSDS500/data/groundTruth/train/22090.mat'
        # return '../BSDS500/data/groundTruth/train/15004.mat'
        # return '../BSDS500/data/groundTruth/test/393035.mat'  # 论文中船
        return '../BSDS500/data/groundTruth/test/384022.mat'  # 论文中船

    def show_three(self):
        path = '../image/leaf.jpg'
        mat = cv2.imread(path)
        print(mat)
        cv2.imshow('leaf', mat)
        cv2.waitKey(0)

# ip = ImagePath()
# ip.show_three()
