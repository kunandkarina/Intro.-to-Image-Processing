import cv2
import numpy as np

def nearest_neighbor_interpolation():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    new_img = np.zeros((h*2, w*2, c), dtype=np.uint8)
    new_h, new_w = new_img.shape[:2]
    for i in range(new_h):
        for j in range(new_w):
            x = int(i / 2)
            y = int(j / 2)
            new_img[i, j] = img[x, y]
    cv2.imshow('Nearest Neighbor Interpolation', new_img)
    # cv2.imwrite('n.jpg', new_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    nearest_neighbor_interpolation()