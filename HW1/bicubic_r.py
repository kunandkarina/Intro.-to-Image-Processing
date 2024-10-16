import cv2
import numpy as np

def bicubic_rotate():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    center_x = w // 2
    center_y = h // 2

    angle = np.radians(30)
    img = img.astype(np.float32)
    rotated_img = np.zeros_like(img, dtype=np.int32)

    for i in range(h):
        for j in range(w):
            x_s = i - center_x
            y_s = j - center_y
            x_r = x_s * np.cos(angle) - y_s * np.sin(angle)
            y_r = x_s * np.sin(angle) + y_s * np.cos(angle)
            x_r = int(x_r) + center_x
            y_r = int(y_r) + center_y
            if 0 <= x_r < h-1 and 0 <= y_r < w-1:
                tp = []
                for k in range(-1,3):
                    x0 = int(x_r) + k
                    if x0 < 0:
                        x0 = 0
                    elif x0 >= h:
                        x0 = h-1
                    p = []
                    for l in range(-1,3):
                        y0 = int(y_r) + l
                        if y0 < 0:
                            y0 = 0
                        elif y0 >= w:
                            y0 = w-1
                        p.append(img[x0, y0])
                    f_y = (y_r - int(y_r))
                    tp.append((-0.5*p[0]+1.5*p[1]-1.5*p[2]+0.5*p[3])*f_y**3 + (p[0]-2.5*p[1]+2*p[2]-0.5*p[3])*f_y**2 + (-0.5*p[0]+0.5*p[2])*f_y + p[1])
                f_x = (x_r - int(x_r))
                rotated_img[i, j] = (-0.5*tp[0]+1.5*tp[1]-1.5*tp[2]+0.5*tp[3])*f_x**3 + (tp[0]-2.5*tp[1]+2*tp[2]-0.5*tp[3])*f_x**2 + (-0.5*tp[0]+0.5*tp[2])*f_x + tp[1]

    rotated_img = np.clip(rotated_img, 0, 255)
    rotated_img = rotated_img.astype(np.uint8)
    cv2.imshow('Bicubic Rotated Image', rotated_img)
    # cv2.imwrite('bcr.jpg', rotated_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    bicubic_rotate()