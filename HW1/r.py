import cv2
import numpy as np

img = cv2.imread('building.jpg')
h, w, c = img.shape
center_x = w // 2
center_y = h // 2

degree = 37
angle = np.radians(degree)

x = 3 - center_x
y = 4 - center_y
print(x, y)
x_r = int(x * np.cos(-angle) + y * np.sin(-angle))
y_r = int(-x * np.sin(-angle) + y * np.cos(-angle))
print(x_r, y_r)
x_r += center_x
y_r += center_y
print(x_r, y_r)