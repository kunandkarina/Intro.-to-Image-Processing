import cv2
import numpy as np
from PIL import Image, ImageFilter


for i in range(1, 16):
    # img = cv2.imread('hat_result/' + str(i) + '_HAT_GAN_Real_SRx4.png')
    # gaussian_3 = cv2.GaussianBlur(img, (3, 3), 0)
    # unsharp_image = cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0)
    # cv2.imwrite(f'hat_filter/{i}.png', gaussian_3)
    img = cv2.imread('hat_result/' + str(i) + '_HAT_GAN_Real_SRx4.png')
    img1 = cv2.resize(img, (512, 512));
    cv2.imwrite(f'result/{i}.png', img1)
    

