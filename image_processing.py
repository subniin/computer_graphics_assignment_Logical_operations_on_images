import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

img_path = 'he_image.jpg'
img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# HE
img_he = cv2.equalizeHist(img_orig)

# AHE (clipLimit을 매우 높게 주어 제한을 없앰)
ahe = cv2.createCLAHE(clipLimit=255.0, tileGridSize=(8, 8))
img_ahe = ahe.apply(img_orig)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_orig)

# 평가지표 계산 함수
def calc_metrics(orig, processed):
    m = mse(orig, processed)
    p = psnr(orig, processed)
    s = ssim(orig, processed, data_range=processed.max() - processed.min())
    return m, p, s

# 지표 계산
metrics_he = calc_metrics(img_orig, img_he)
metrics_ahe = calc_metrics(img_orig, img_ahe)
metrics_clahe = calc_metrics(img_orig, img_clahe)

# 시각화 (이미지 배열 및 리스트 준비)
images = [img_orig, img_he, img_ahe, img_clahe]
titles = [
    "Original Image", 
    f"HE\nMSE:{metrics_he[0]:.2f} PSNR:{metrics_he[1]:.2f} SSIM:{metrics_he[2]:.2f}",
    f"AHE\nMSE:{metrics_ahe[0]:.2f} PSNR:{metrics_ahe[1]:.2f} SSIM:{metrics_ahe[2]:.2f}",
    f"CLAHE\nMSE:{metrics_clahe[0]:.2f} PSNR:{metrics_clahe[1]:.2f} SSIM:{metrics_clahe[2]:.2f}"
]

plt.figure(figsize=(12, 16))

for i in range(4):
    # 이미지 출력
    plt.subplot(4, 2, i*2 + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=12)
    plt.axis('off')
    
    # 히스토그램 출력
    plt.subplot(4, 2, i*2 + 2)
    # 히스토그램 계산 (0~256 범위)
    hist = cv2.calcHist([images[i]], [0], None, [256], [0, 256])
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.title("Histogram")

plt.tight_layout()
plt.show()