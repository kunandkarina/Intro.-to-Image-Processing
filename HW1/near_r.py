import cv2
import numpy as np

def nearest_rotate():
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
            if 0 <= x_r < h and 0 <= y_r < w:
                rotated_img[i, j] = img[x_r, y_r]
    cv2.imshow('Rotated Image', rotated_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    nearest_rotate()