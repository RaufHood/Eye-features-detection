import cv2
import matplotlib.pyplot as plt 
import numpy as np

def get_largest_detection(detections):
    """Return the largest detected region based on area."""
    if not detections.any():
        return None
    return max(detections, key=lambda rect: rect[2] * rect[3])

def crop_image(largest_eye, image):
    if largest_eye is not None:
        ex, ey, ew, eh = largest_eye
        # Draw a rectangle around the detected eye
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # Crop the image to get only the eye
        eye_img = image[ey:ey+eh, ex:ex+ew]
        # plt.imshow(eye_img)
        # plt.show()
        #im = Image.fromarray(eye_img)
        #im.save("eyee.jpg")
    else:
        print("No eye detected")
    return eye_img


def morphological_reconstruction(marker, mask):
    """Perform morphological reconstruction on the given marker using the mask."""
    kernel = np.ones((3, 3), np.uint8)
    while True:
        dilation = cv2.dilate(marker, kernel, iterations=1)
        plt.imshow(dilation)
        temp = cv2.min(dilation, mask)
        if np.array_equal(marker, temp):
            return marker
        marker = temp

def remove_overlapping_circles(circles):
    """Remove overlapping circles and keep the ones with the highest accumulator values."""
    if circles is None:
        return None

    circles = sorted(circles, key=lambda x: -x[2])  # Sort by radius (largest first)
    valid_circles = []
    for circle in circles:
        x, y, r = circle
        overlapping = False
        for valid_circle in valid_circles:
            vx, vy, vr = valid_circle
            dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            if dist < r + vr:  # Overlapping condition
                overlapping = True
                break
        if not overlapping:
            valid_circles.append(circle)
    return valid_circles