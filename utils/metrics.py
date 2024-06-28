import numpy as np
from utils import helpers


def accuracy(segmented, mask):
    correct_predictions = 0
    total_predictions = 0
    for label, color_rgb in helpers.LABEL_COLORS.items():
        true_positives = np.sum((segmented == color_rgb) & (mask == color_rgb))
        true_negatives = np.sum((segmented != color_rgb) & (mask != color_rgb))
        false_positives = np.sum((segmented == color_rgb) & (mask != color_rgb))
        false_negatives = np.sum((segmented != color_rgb) & (mask == color_rgb))
        correct_predictions += true_positives + true_negatives
        total_predictions += true_positives + true_negatives + false_positives + false_negatives

    return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0


def precision(segmented, mask):
    precisions = []
    for label, color_rgb in helpers.LABEL_COLORS.items():
        true_positives = np.sum((segmented == color_rgb) & (mask == color_rgb))
        predicted_positives = np.sum(segmented == color_rgb)
        precision_score = true_positives / predicted_positives if predicted_positives > 0 else 0
        precisions.append(precision_score)

    return np.mean(precisions)


def recall(segmented, mask):
    recalls = []
    for label, color_rgb in helpers.LABEL_COLORS.items():
        true_positives = np.sum((segmented == color_rgb) & (mask == color_rgb))
        actual_positives = np.sum(mask == color_rgb)
        recall_score = true_positives / actual_positives if actual_positives > 0 else 0
        recalls.append(recall_score)

    return np.mean(recalls)


def intersection_over_union(segmented, mask):
    intersections = 0
    unions = 0
    for label, color_rgb in helpers.LABEL_COLORS.items():
        intersection = np.sum((segmented == color_rgb) & (mask == color_rgb))
        union = np.sum((segmented == color_rgb) | (mask == color_rgb))
        intersections += intersection
        unions += union

    return intersections / unions if unions > 0 else 0


def dice_coefficient(segmented, mask):
    dice_scores = []
    for label, color_rgb in helpers.LABEL_COLORS.items():
        intersection = np.sum((segmented == color_rgb) & (mask == color_rgb))
        seg_size = np.sum(segmented == color_rgb)
        mask_size = np.sum(mask == color_rgb)
        dice_score = 2.0 * intersection / (seg_size + mask_size)
        dice_scores.append(dice_score)

    return np.mean(dice_scores)
