import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

def rgb_to_lab(rgb_img):
    lab_img = color.rgb2lab(rgb_img)
    return lab_img


def delta_e_ciede2000(lab1, lab2, k_L=1, k_C=1, k_H=1):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    delta_L = L2 - L1
    delta_a = a2 - a1
    delta_b = b2 - b1

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2

    delta_C_prime = C2 - C1

    delta_h_rad = np.arctan2(b2, a2) - np.arctan2(b1, a1)
    delta_h_deg = np.rad2deg(delta_h_rad)
    delta_h_prime = np.where(
        np.abs(delta_h_deg) <= 180,
        delta_h_deg,
        np.where(delta_h_deg > 180, delta_h_deg - 360, delta_h_deg + 360)
    )

    delta_H_prime = 2 * np.sqrt(C1 * C2) * np.sin(np.deg2rad(delta_h_prime) / 2)

    L_avg = (L1 + L2) / 2
    C7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C7 / (C7 + 25 ** 7)))
    SL = 1 + (0.015 * (L_avg - 50) ** 2) / np.sqrt(20 + (L_avg - 50) ** 2)
    SC = 1 + 0.045 * C_avg
    SH = 1 + 0.015 * C_avg * np.cos(np.deg2rad(L_avg - 30))

    delta_theta = 30 * np.exp(-((L_avg - 275) / 25) ** 2)
    RT = -2 * np.sqrt(C7 / (C7 + 25 ** 7)) * np.sin(np.deg2rad(2 * delta_theta))

    term1 = delta_L / (k_L * SL)
    term2 = delta_C_prime / (k_C * SC)
    term3 = delta_H_prime / (k_H * SH)
    term4 = RT * term2 * term3
    pixel_delta_e = np.sqrt(term1 ** 2 + term2 ** 2 + term3 ** 2 + term4)
    
    valid_delta_e = pixel_delta_e[pixel_delta_e <= np.percentile(pixel_delta_e, 99.9)]
    return np.mean(valid_delta_e)