import numpy as np
import cv2

img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

img_h, img_w = img.shape
kernel_h, kernel_w = kernel.shape

pad_h = (kernel_h-1) // 2
pad_w = (kernel_w-1) // 2

pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
n_img = np.zeros(img.shape)

for i in range(img_h):
    for j in range(img_w):
        n_img[i, j] = np.sum(pad[i:i+kernel_h, j:j+kernel_w]*kernel)

laplacian = np.clip(n_img, 0, 255)
c = -1
g = img + c * laplacian
g_clip = np.clip(g, 0, 255)
# print(g_clip.dtype)

cv2.imshow('spatial_k4', g_clip.astype(np.uint8))
# cv2.imwrite('spatial_k4.jpg', g_clip)
cv2.waitKey(0)
cv2.destroyAllWindows()
