import cv2
import numpy as np

def bicubic_interpolation():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    img = img.astype(np.float32)
    new_img = np.zeros((h*2, w*2, c), dtype=np.int32)
    new_h, new_w = new_img.shape[:2]
    for i in range(new_h):
        for j in  range(new_w):
            x1 = int(i/2)
            y1 = int(j/2)
            tp = []
            for k in range(-1,3):
                x = x1 + k
                if x < 0:
                    x = 0
                elif x >= h:
                    x = h-1
                p = []
                for l in range(-1,3):
                    y = y1 + l
                    if y < 0:
                        y = 0
                    elif y >= w:
                        y = w-1
                    p.append(img[x, y])
                
                f_y = (j/2) - y1
                tp.append((-0.5*p[0]+1.5*p[1]-1.5*p[2]+0.5*p[3])*f_y**3 + (p[0]-2.5*p[1]+2*p[2]-0.5*p[3])*f_y**2 + (-0.5*p[0]+0.5*p[2])*f_y + p[1])
            f_x = (i/2) - x1
            new_img[i, j] = (-0.5*tp[0]+1.5*tp[1]-1.5*tp[2]+0.5*tp[3])*f_x**3 + (tp[0]-2.5*tp[1]+2*tp[2]-0.5*tp[3])*f_x**2 + (-0.5*tp[0]+0.5*tp[2])*f_x + tp[1]

    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)
    cv2.imshow('Bicubic Interpolation', new_img)
    # cv2.imwrite('bc.jpg', new_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    bicubic_interpolation()