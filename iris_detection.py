import cv2
import matplotlib.pyplot as plt
import numpy as np
#from pupil_detectors import Detector2D
#from PIL import Image
from utils.detection import *

# 1 Load the image
img = cv2.imread("images/2.jpeg") # image (3 channels)
image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # image 1 channels
# pixels = np.array(image) # pixels is an array of the image
# h, w = pixels.shape
#2. Plot the image
# print(img.shape, image.shape)

# 2. Detect eye
# Create an instance of the cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# detect eye
eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.05)
largest_eye = get_largest_detection(eyes)
eye_img = crop_image(largest_eye, img)
# gray = cv2.equalizeHist(image)
# TO DO:
# detect other eye as well

# 3. extract red channel
red_channel = eye_img[:, :, 2]
# create a binary mask to detect iris
_, binary_mask = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 200 is a sample threshold, adjust as needed

# Perform morphological reconstruction
# erosion
marker = cv2.erode(red_channel, np.ones((3, 3), np.uint8), iterations=10) 
# dilation
reconstructed_red_channel = morphological_reconstruction(marker, red_channel) # binary mask

# plt.imshow(reconstructed_red_channel)
# plt.show()

circles = cv2.HoughCircles(reconstructed_red_channel, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=40, minRadius=15, maxRadius=50)
print("circles ", circles)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    circles = remove_overlapping_circles(circles)
    for (x, y, r) in circles:
        cv2.circle(reconstructed_red_channel, (x, y), r, (0, 255, 0), 4)

    # Display the image with detected circles
    plt.imshow(reconstructed_red_channel)
    plt.show()
    plt.title('Detected iris')