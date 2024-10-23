import cv2
import matplotlib.pyplot as plt 
import numpy as np

def get_largest_detection(detections):
    """Return the largest detected region based on area."""
    if not detections.any():
        return None
    return max(detections, key=lambda rect: rect[2] * rect[3])

def crop_image(largest_eye, image):
    """Return a cropped image based on the pixel location of largest_eye."""
    if largest_eye is not None:
        ex, ey, ew, eh = largest_eye
        # Draw a rectangle around the detected eye
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_img = image[ey:ey+eh, ex:ex+ew]
        
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
    for circle in circles[0]:
        #print(circle[0])
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

def extract_circle_opencv(image, circle):
    "Return an image after applying a circular mask."
    x, y, r = circle
    # Create a mask with the same dimensions as the image, initialized to zero (black)
    mask = np.zeros_like(image)
    # Draw a white circle (value 255) in the mask at position (x, y) with radius r
    cv2.circle(mask, (x, y), r, 255, thickness=-1)
    # Apply the mask using bitwise AND
    result_image = cv2.bitwise_and(image, mask)
    
    return result_image

def approximate_pupil_circle(binary_mask):
    """Approximate the pupil as a circle using the binary mask."""
    # Compute the centroid of the binary mask
    M = cv2.moments(binary_mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Compute the average distance of mask pixels to the centroid
    y_indices, x_indices = np.where(binary_mask > 0)
    distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    avg_distance = np.mean(distances)

    return (cx, cy, int(avg_distance))

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


