import cv2
import numpy as np

def bilinear_rotate():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    center_x = w // 2
    center_y = h // 2

    angle = np.radians(30)
    rotated_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            x_s = i - center_x
            y_s = j - center_y
            x_r = x_s * np.cos(angle) - y_s * np.sin(angle)
            y_r = x_s * np.sin(angle) + y_s * np.cos(angle)
            x_r = int(x_r) + center_x
            y_r = int(y_r) + center_y
            if 0 <= x_r < h-1 and 0 <= y_r < w-1:
                x0 = int(x_r)
                x1 = x0 + 1
                y0 = int(y_r)
                y1 = y0 + 1
                x_t = x_r - x0
                y_t = y_r - y0
                in_1 = img[x0, y0] * (1-x_t) + img[x0, y1] * x_t
                in_2 = img[x1, y0] * (1-x_t) + img[x1, y1] * x_t
                rotated_img[i, j] = in_1 * (1-y_t) + in_2 * y_t

    cv2.imshow('Bilinear Rotated Image', rotated_img)
    # cv2.imwrite('blr.jpg', rotated_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    bilinear_rotate()