import cv2
import numpy as np

def nearest_rotate():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    center_x = w // 2
    center_y = h // 2

    angle = np.radians(30)
    rotated_img = np.zeros_like(img, dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            x_shifted = x - center_x
            y_shifted = y - center_y
            x_rotated = int(x_shifted * np.cos(-angle) - y_shifted * np.sin(-angle))
            y_rotated = int(x_shifted * np.sin(-angle) + y_shifted * np.cos(-angle))
            x_rotated += center_x
            y_rotated += center_y

            if 0 <= x_rotated < w-1 and 0 <= y_rotated < h-1:
                rotated_img[y, x] = img[y_rotated, x_rotated]
    
    cv2.imshow('Rotated Image', rotated_img)
    # cv2.imwrite('nr.jpg', rotated_img)
    cv2.waitKey(0)

def bilinear_rotate():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    center_x = w // 2
    center_y = h // 2

    angle = np.radians(30)
    rotated_img = np.zeros_like(img, dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            # 將像素座標平移至原點
            x_shifted = x - center_x
            y_shifted = y - center_y
            x_rotated = int(x_shifted * np.cos(-angle) - y_shifted * np.sin(-angle))
            y_rotated = int(x_shifted * np.sin(-angle) + y_shifted * np.cos(-angle))
            x_rotated += center_x
            y_rotated += center_y

            if 0 <= x_rotated < w-1 and 0 <= y_rotated < h-1:
                x0 = int(x_rotated)
                x1 = min(x0 + 1, w-1)
                y0 = int(y_rotated)
                y1 = min(y0 + 1, h-1)
                x_t = x_rotated - x0
                y_t = y_rotated - y0
                in_1 = img[y0, x0] * (1-x_t) + img[y0, x1] * x_t
                in_2 = img[y1, x0] * (1-x_t) + img[y1, x1] * x_t
                rotated_img[y, x] = in_1 * (1-y_t) + in_2 * y_t

    cv2.imshow('Bilinear Rotated Image', rotated_img)
    # cv2.imwrite('blr.jpg', rotated_img)
    cv2.waitKey(0)

def bicubic_rotate():
    img = cv2.imread('building.jpg')
    h, w, c = img.shape
    center_x = w // 2
    center_y = h // 2

    angle = np.radians(30)
    img = img.astype(np.float32)
    rotated_img = np.zeros_like(img, dtype=np.int32)

    for x in range(w):
        for y in range(h):
            x_shifted = x - center_x
            y_shifted = y - center_y
            x_rotated = int(x_shifted * np.cos(-angle) - y_shifted * np.sin(-angle))
            y_rotated = int(x_shifted * np.sin(-angle) + y_shifted * np.cos(-angle))
            x_rotated += center_x
            y_rotated += center_y
            if 0 <= x_rotated < w-1 and 0 <= y_rotated < h-1:
                tp = []
                for i in range(-1,3):
                    x0 = int(x_rotated) + i
                    if x0 < 0:
                        x0 = 0
                    elif x0 >= w:
                        x0 = w-1
                    p = []
                    for j in range(-1,3):
                        y0 = int(y_rotated) + j
                        if y0 < 0:
                            y0 = 0
                        elif y0 >= h:
                            y0 = h-1
                        p.append(img[y0, x0])
                    f_y = (y_rotated - int(y_rotated))
                    tp.append((-0.5*p[0]+1.5*p[1]-1.5*p[2]+0.5*p[3])*f_y**3 + (p[0]-2.5*p[1]+2*p[2]-0.5*p[3])*f_y**2 + (-0.5*p[0]+0.5*p[2])*f_y + p[1])
                f_x = (x_rotated - int(x_rotated))
                rotated_img[y, x] = (-0.5*tp[0]+1.5*tp[1]-1.5*tp[2]+0.5*tp[3])*f_x**3 + (tp[0]-2.5*tp[1]+2*tp[2]-0.5*tp[3])*f_x**2 + (-0.5*tp[0]+0.5*tp[2])*f_x + tp[1]

    rotated_img = np.clip(rotated_img, 0, 255)
    rotated_img = rotated_img.astype(np.uint8)
    cv2.imshow('Bicubic Rotated Image', rotated_img)
    # cv2.imwrite('bcr.jpg', rotated_img)
    cv2.waitKey(0)
                

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
    nearest_rotate()
    bilinear_rotate()
    bicubic_rotate()
    nearest_neighbor_interpolation()
    bilinear_interpolation()
    bicubic_interpolation()