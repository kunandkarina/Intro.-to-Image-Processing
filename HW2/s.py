import cv2
import numpy as np
import matplotlib.pyplot as plt

img_s = cv2.imread('Q2_source.jpg')
img_r = cv2.imread('Q2_reference.jpg')

g_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY) 
g_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

h_s = np.zeros(256)
s_r, s_c = g_s.shape
h_r = np.zeros(256)
r_r, r_c = g_r.shape

for i in range(s_r):
    for j in range(s_c):
        h_s[g_s[i, j]] += 1

for i in range(r_r):
    for j in range(r_c):
        h_r[g_r[i, j]] += 1

p_s = np.zeros(256)
p_r = np.zeros(256)
for i in range(256):
    p_s[i] = h_s[i] / (s_r * s_c)
    p_r[i] = h_r[i] / (r_r * r_c)

ss = np.zeros(256)
sr = np.zeros(256)
cdf_s = np.zeros(256)
cdf_r = np.zeros(256)

tp_s = 0
tp_r = 0
for i in range(256):
    tp_s += p_s[i]
    ss[i] = round(255 * tp_s)
    cdf_s[i] = tp_s
    tp_r += p_r[i]
    sr[i] = round(255 * tp_r)
    cdf_r[i] = tp_r

tar_map = np.zeros(256, dtype=np.uint8)
for i in range(256):
    diff = abs(sr - ss[i])
    tar_map[i] = np.argmin(diff)

for i in range(s_r):
    for j in range(s_c):
        g_s[i,j] = tar_map[g_s[i,j]]

cv2.imshow("histogram_spec", g_s)
cv2.imwrite("histogram_spec.jpg", g_s)
cv2.waitKey(0)
cv2.destroyAllWindows()


