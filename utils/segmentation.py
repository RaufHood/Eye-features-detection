import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from utils.detection import remove_overlapping_circles, calculate_circularity

def detect_iris(image):
    # Step 1: Read the input image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for illumination normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 4: Apply Gaussian blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Step 5: Perform Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=10)#150

    # Step 6: Detect circles using HoughCircles
    circles = cv2.HoughCircles(edges, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1, 
                               minDist=30, 
                               param1=100, 
                               param2=30, 
                               minRadius=20, 
                               maxRadius=100)
    circles = remove_overlapping_circles(circles)
    #print("circles ", circles)
    #plt.imshow(edges)
    #plt.show()
    # Step 7: Draw detected circles (iris) on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        rr = [r for _, _, r in circles]
        index = np.argmax(rr)
        circle = circles[index]
        circles = [circle]
        for i in circles: #[0, :]
            # Draw the outer circle
            cv2.circle(gray, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # Save the iris center for pupil detection
            iris_center = (i[0], i[1])
            iris_radius = i[2]
        print("Iris detected successfully.")
    else:
        print("No iris detected.")
        return
    

    return gray, blurred, edges, iris_center, iris_radius


def detect_pupil_with_circle(binary_mask, iris_center, iris_radius):
    # Step 1: Create a circular mask based on the detected iris
    mask_shape = binary_mask.shape
    iris_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.circle(iris_mask, iris_center, iris_radius, 255, thickness=-1)
    # Step 2: Apply the iris mask to the binary mask
    masked_binary = cv2.bitwise_and(binary_mask, iris_mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    masked_binary = clahe.apply(masked_binary)
    plt.imshow(masked_binary)
    plt.show()
    # Step 3: Find contours in the masked binary image
    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None

    # Step 4: Find the largest contour, which should correspond to the pupil region
    largest_contour = max(contours, key=cv2.contourArea)
    pupil_area = cv2.contourArea(largest_contour)

    #######
    circularity_threshold = 0.5 # Adjust as needed (closer to 1 means more circular)
    possible_pupil_contours = [cnt for cnt in contours if calculate_circularity(cnt) > circularity_threshold]

    if not possible_pupil_contours:
        print("No circular contour found.")
        return None
    else:
        print('Contour found!')
        print('contour', contours)
        print('pupil area', pupil_area)
        cv2.drawContours(masked_binary, contours, -1, (0,255,0), 3)
        plt.imshow(masked_binary)
        plt.show()

    # Choose the largest contour from the filtered results
    largest_contour = max(possible_pupil_contours, key=cv2.contourArea)
    pupil_area = cv2.contourArea(largest_contour)
    #######

    # Step 5: Approximate a circle that covers 90% of the largest contour's area
    # Calculate the radius of a circle that would cover 90% of the pupil area
    target_area = pupil_area * 0.9
    target_radius = int(np.sqrt(target_area / np.pi))

    # Get the minimum enclosing circle for the largest contour
    (x, y), enclosing_radius = cv2.minEnclosingCircle(largest_contour)

    # Limit the radius to be the smaller of the calculated target radius or enclosing radius
    final_radius = min(target_radius, int(enclosing_radius))

    # Make sure the final circle stays within the iris boundary
    if final_radius > iris_radius:
        final_radius = iris_radius

    pupil_center = (int(x), int(y))
    return pupil_center, final_radius

def bounding_box_segmentation(image, predictor):
    # Estimate eye location (you may need to adjust these values)
    h, w = image.shape[:2]
    box = np.array([w, h, w, h])
    masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
    return masks[0]

def calculate_metrics(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    # Intersection over Union
    iou = jaccard_score(gt_flat, pred_flat, average='binary')
    # Dice coefficient
    dice = f1_score(gt_flat, pred_flat,average='binary')
    # Pixel Accuracy
    pixel_accuracy = np.sum(pred_mask == gt_mask) / gt_mask.size

    return {"IOU": iou, 'dice': dice, 'acc': pixel_accuracy}

def evaluate_segmentation(pred_mask, gt_mask ):
    #pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)/255
    #gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)/255
    
    class_labels = [0, 1, 2, 3]
    metrics_per_class = {}
    for class_label in class_labels:
        pred_binary = (pred_mask == class_label).astype(np.uint8)
        gt_binary = (gt_mask ==class_label).astype(np.uint8)

        metrics = calculate_metrics(pred_binary, gt_binary)
        metrics_per_class[f'class_{class_label}'] = metrics

    print("Segmentation metrics per class:")
    for class_label, metrics in metrics_per_class.items():
        print(f"{class_label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return metrics_per_class

