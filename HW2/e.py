import cv2
import numpy as np

img = cv2.imread('Q1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r, c = gray.shape

h = np.zeros(256)
for i in range(r):
    for j in range(c):
        h[gray[i, j]] += 1

p = np.zeros(256)
for i in range(256):
    p[i] = h[i] / (r * c)

s = np.zeros(256)
tp = 0
for i in range(256):
    tp += p[i]
    s[i] = round(255 * tp)


ans = np.zeros((r,c,3), np.uint8)
for i in range(r):
    for j in range(c):
        ans[i,j] = s[gray[i,j]]

cv2.imshow('ans', ans)
cv2.imwrite('histogram_equal.jpg', ans)
cv2.waitKey(0)
cv2.destroyAllWindows()
