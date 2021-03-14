import cv2
import skimage.feature as sk
import numpy as np

# Read an image from the disk and convert it into grayscale
image = cv2.imread("images/nature.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate GLCM of the grayscale image
glcm = sk.greycomatrix(image, [2], [0, np.pi / 2])

# Calculate Contrast
contrast = sk.greycoprops(glcm)
print("Contrast:", contrast)

# Calculate ‘dissimilarity’
dissimilarity = sk.greycoprops(glcm, prop='dissimilarity')
print("Dissimilarity: ", dissimilarity)

# Calculate ‘homogeneity’
homogeneity = sk.greycoprops(glcm, prop='homogeneity')
print("Homogeneity: ", homogeneity)

# Calculate ‘ASM’
ASM = sk.greycoprops(glcm, prop='ASM')
print("ASM: ", ASM)

# Calculate ‘energy’
energy = sk.greycoprops(glcm, prop='energy')
print("Energy: ", energy)

# Calculate ‘correlation’
correlation = sk.greycoprops(glcm, prop='correlation')
print("Correlation: ", correlation)
