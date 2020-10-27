import skimage
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

class measure:
    def F(self, gt, my_bin):
        tmp = my_bin - gt
        FP = sum(sum(tmp == 255))
        FN = sum(sum(tmp == -255))
        TP = sum(sum((my_bin == gt) & (gt == 255)))
        p = TP / (TP + FP)
        r = TP / (TP + FN)

        belt2 = 0.1

        Fmeasure = ((1 + belt2 ) * p * r) / (belt2 * (p + r))
        return p, r, Fmeasure


    def PSNR(self, ori_mat, blur_mat):
        psnr = skimage.measure.compare_psnr(ori_mat, blur_mat, 255)
        return psnr

    '''
    下面三个方法是计算 SSIM
    
    '''
    def matlab_style_gauss2D(self, shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def filter2(self, x, kernel, mode='same'):
        return convolve2d(x, np.rot90(kernel, 2), mode=mode)
    def compute_ssim(self, im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

        if not im1.shape == im2.shape:
            raise ValueError("Input Imagees must have the same dimensions")
        if len(im1.shape) > 2:
            raise ValueError("Please input the images with 1 channel")

        M, N = im1.shape
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = self.matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))

        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)

        mu1 = self.filter2(im1, window, 'valid')
        mu2 = self.filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.filter2(im1 * im1, window, 'valid') - mu1_sq
        sigma2_sq = self.filter2(im2 * im2, window, 'valid') - mu2_sq
        sigmal2 = self.filter2(im1 * im2, window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return np.mean(np.mean(ssim_map))

