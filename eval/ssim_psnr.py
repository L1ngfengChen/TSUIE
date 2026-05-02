import argparse
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pcqi import PCQI
from deltaE import delta_e_ciede2000, rgb_to_lab

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


im_path = '/data1/chenlingfeng/project/TSUIE_code/output/'
re_path = '/data1/chenlingfeng/reproduction/dataset/U90/U90_ori/'

supported_formats = ['.jpg', '.png', '.bmp', '.jpeg', '.tiff', '.webp']

avg_psnr = 0
avg_ssim = 0
avg_pcqi = 0
avg_mse = 0
avg_deltaE = 0

n = 0

for filename in os.listdir(im_path):
    input_file_path = os.path.join(im_path, filename)
    base_filename, ext = os.path.splitext(filename)

    reference_file_path = None
    for fmt in supported_formats:
        candidate_path = os.path.join(re_path, base_filename + fmt)
        if os.path.exists(candidate_path):
            reference_file_path = candidate_path
            break
    
    if not reference_file_path:
        print(f"警告：未找到 {base_filename} 的参考图像，支持格式：{', '.join(supported_formats)}")
        continue

    n = n + 1
    im1 = cv2.imread(input_file_path)
    im2 = cv2.imread(reference_file_path)

    (h, w, c) = im1.shape
    im1 = cv2.resize(im1, (256,256))
    im2 = cv2.resize(im2, (256,256))

    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im1_rgb_norm = im1_rgb.astype(np.float32) / 255.0
    im2_rgb_norm = im2_rgb.astype(np.float32) / 255.0
    im1_lab = rgb_to_lab(im1_rgb_norm)
    im2_lab = rgb_to_lab(im2_rgb_norm)
    score_delta_e = delta_e_ciede2000(im1_lab, im2_lab)

    score_psnr = psnr(im1, im2)
    score_ssim = ssim(im1, im2, channel_axis=2, win_size=7)

    im1 = im1 / 255.0
    im2 = im2 / 255.0
    score_mse = mse(im1, im2)

    ref = cv2.imread(reference_file_path, 0)
    raw = cv2.imread(input_file_path, 0)
    ref = cv2.resize(ref, (256, 256))
    raw = cv2.resize(raw, (256, 256))
    ref = np.array(ref.tolist())
    raw = np.array(raw.tolist())
    score_pcqi, _ = PCQI(ref,raw)


    avg_psnr += score_psnr
    avg_ssim += score_ssim
    avg_pcqi += score_pcqi
    avg_mse += score_mse
    avg_deltaE += score_delta_e


avg_psnr = avg_psnr / n
avg_ssim = avg_ssim / n
avg_pcqi /= n
avg_mse /= n
avg_deltaE /= n


print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
print("===> Avg.PCQI: {:.4f}".format(avg_pcqi))
print("===> Avg.MSE: {:.4f}".format(avg_mse))
print(f"===> Avg.DeltaE (CIEDE2000): {avg_deltaE:.4f} ")