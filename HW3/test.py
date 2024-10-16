import numpy as np
import cv2

# Read the input image
img = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier transform
f_img = np.fft.fft2(img)

# Define Laplacian filter in the frequency domain
laplacian_filter = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

# Perform Fourier transform on the Laplacian filter
f_laplacian_filter = np.fft.fft2(laplacian_filter, s=img.shape)

# Apply the Laplacian filter in the frequency domain
f_filtered_img = f_img * f_laplacian_filter

# Perform inverse Fourier transform
filtered_img = np.fft.ifft2(f_filtered_img).real

# Normalize the filtered image to the range [0, 255]
filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

# Display the filtered image
cv2.imshow('Filtered Image', filtered_img)
cv2.imwrite('filtered_image.jpg', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
