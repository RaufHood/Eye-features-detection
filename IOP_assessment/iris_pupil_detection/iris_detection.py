import cv2
import matplotlib.pyplot as plt
import numpy as np
#from pupil_detectors import Detector2D
#from PIL import Image

img = cv2.imread("images/2.jpeg") # import image
image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # image has 3 channels -> to gray
pixels = np.array(image) # pixels is an array of the image
# h, w = pixels.shape
plt.plot(pixels)

