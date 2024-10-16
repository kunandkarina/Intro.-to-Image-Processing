import cv2
import numpy as np

img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)
img = img / 255
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

P, Q = fshift.shape
H = np.zeros((P,Q), dtype=np.float32)
for u in range(P):
    for v in range(Q):
        H[u, v] = -4 * np.pi**2 * ((u - P/2)**2 + (v - Q/2)**2)

Lap = H * fshift
Lap = np.fft.ifftshift(Lap)
Lap = np.real(np.fft.ifft2(Lap))
old_range = np.max(Lap) - np.min(Lap)
new_range = 1 - (-1)
LapScale = (((Lap - np.min(Lap)) * new_range) / old_range) + -1

c = -1
g = img + c * LapScale
g = np.clip(g, 0, 1)

g = (g * 255).astype(np.uint8)

cv2.imshow('moon_f', g)
# cv2.imwrite('moon_f.jpg', g)
cv2.waitKey(0)
cv2.destroyAllWindows()
