import cv2
import numpy as np

def bilinear_interpolation():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape

    sh = 2
    sw = 2

    new_img = np.zeros((h*sh,w*sw,3), np.uint8)
    hz = new_img.shape[0]
    wz = new_img.shape[1]
    for i in range(hz):
        for j in range(wz):
            p1_i = int(i/sh)
            p1_j = int(j/sw)
            p1 = img[p1_i, p1_j]
            p2 = img[p1_i, min(p1_j+1, w-1)]
            p3 = img[min(p1_i+1, h-1), p1_j]
            p4 = img[min(p1_i+1, h-1), min(p1_j+1, w-1)]

            q1 = (p1_j+1 - j/sw)*p1 + (j/sw - p1_j)*p2
            q2 = (p1_j+1 - j/sw)*p3 + (j/sw - p1_j)*p4
            p = (i/sh - p1_i)*q2 + (p1_i+1 - i/sh)*q1
            new_img[i,j] = p
    cv2.imshow('Bilinear Interpolation', new_img)
    # cv2.imwrite('b.jpg', new_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    bilinear_interpolation()