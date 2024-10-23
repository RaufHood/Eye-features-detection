import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def calculate_metrics(pred_mask, gt_mask):
    "Returns the calculated metrics IoU, Dice, Accuracy."
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
    "Returns metrics calculated for each class."
    class_names = {
        0: 'Background',
        1: 'Sclera',
        2: 'Iris',
        3: 'Pupil'}
    
    class_labels = [0, 1, 2, 3]
    metrics_per_class = {}
    for class_label in class_labels:
        pred_binary = (pred_mask == class_label).astype(np.uint8)
        gt_binary = (gt_mask ==class_label).astype(np.uint8)

        metrics = calculate_metrics(pred_binary, gt_binary)
        metrics_per_class[f'{class_names[class_label]}'] = metrics

    print("Segmentation metrics per class:")
    for class_label, metrics in metrics_per_class.items():
        print(f"{class_label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return metrics_per_class

