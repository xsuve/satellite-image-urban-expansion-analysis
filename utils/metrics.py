import numpy as np


def accuracy(prediction, mask):
    correct = np.sum(prediction == mask)
    total = np.size(prediction)

    acc = np.mean(correct / total) * 100

    return acc


def intersection_over_union(prediction, mask):
    intersection = np.sum(prediction * mask)
    union = np.sum(prediction) + np.sum(mask) - intersection

    iou = np.mean(intersection / union)

    return iou


def dice_coefficient(prediction, mask):
    intersection = np.sum(prediction * mask)
    total = np.sum(prediction) + np.sum(mask)

    dice = np.mean(2 * intersection / total)

    return dice
