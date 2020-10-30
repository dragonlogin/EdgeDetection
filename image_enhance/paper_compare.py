from functions.add_noise.add_noise import add_noise
from functions.measure_method.measure import measure

from image_enhance.paper import paper_wff
from image_enhance.paper import paper_foma
from image_enhance.paper import MDBUTM_2011
from image_enhance.paper import PDBM_2016
pdbm = PDBM_2016()
mdb = MDBUTM_2011()
foma = paper_foma()
an = add_noise()
pp = paper_wff()
ms = measure()
import cv2
class paper_compare(object):
    def test_salt_noise_wanfengfeng(self, mat, noise_mat, num=1):
        '''
        万丰丰论文
        '''
        wf_blur = pp.wanfeng(noise_mat, 10, 30)
        cv2.imshow('wf_blur', wf_blur)

        '''
        wff - PSNR
        '''
        psnr = ms.PSNR(mat, wf_blur)
        print('wf_psnr={}\n'.format(psnr))

        '''
        wff - SSIM
        '''
        ssim = ms.compute_ssim(mat, wf_blur)
        print('wff_ssim={}\n'.format(ssim))

    def test_salt_noise_foma(self, mat, noise_mat):
        '''
        2020-foma论文
        '''


        foma_blur = foma.foma(noise_mat)
        cv2.imshow('foma_blur', foma_blur)
        '''
        foma-PSNR
        '''
        psnr = ms.PSNR(mat, foma_blur)
        print('foma_psnr={}\n'.format(psnr))

        '''
        foma-SSIM
        '''
        ssim = ms.compute_ssim(mat, foma_blur)
        print('foma_ssim={}\n'.format(ssim))

    def test_salt_noise_smf(self, mat, noise_mat):
        '''
        SMF
        '''
        cv2__median = cv2.medianBlur(noise_mat, 3)
        # tl.threeimgtoexcel('./sources/' + 'cv2_img_salt_median', mat, salt_after, cv2_img_salt_median)
        cv2.imshow('cv2_median_blur', cv2__median)
        '''
        SMF - PSNR
        '''
        psnr = ms.PSNR(mat, cv2__median)
        print('cv_median_psnr={}\n'.format(psnr))

        '''
        SMF - SSIM
        '''
        ssim = ms.compute_ssim(mat, cv2__median)
        print('cv_median_ssim={}\n'.format(ssim))

    def test_salt_noise_mdbutm(self, mat, noise_mat):
        '''
        MDBUTM_2011
        '''
        mdb_blur = mdb.mdbutm(noise_mat)
        cv2.imshow('mdb_blur', mdb_blur)
        # tl.threeimgtoexcel('./sources/' + 'cv2_img_salt_median', mat, salt_after, cv2_img_salt_median)
        '''
        MDBUTM_2011 - PSNR
        '''
        psnr = ms.PSNR(mat, mdb_blur)
        print('mdbutm_psnr={}\n'.format(psnr))

        '''
        MDBUTM_2011 - SSIM
        '''
        ssim = ms.compute_ssim(mat, mdb_blur)
        print('mdbutm_ssim={}\n'.format(ssim))

    def test_salt_noise_pdbm(self, mat, noise_mat):
        '''
        PDBM_2016
        '''
        pdbm_blur = pdbm.pdbm(noise_mat)
        cv2.imshow('pdbm_blur',  pdbm_blur)
        '''
        PDBM-2016 - PSNR
        '''
        psnr = ms.PSNR(mat, pdbm_blur)
        print('pdbm_psnr={}\n'.format(psnr))

        '''
        PDBM_2011 - SSIM
        '''
        ssim = ms.compute_ssim(mat, pdbm_blur)
        print('pdbm_ssim={}\n'.format(ssim))


