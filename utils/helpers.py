import cv2
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify

plt.rcParams['toolbar'] = 'None'


LABEL_COLORS = {
    # RGB
    0: [140, 212, 126],  # LAND
    1: [248, 213, 109],  # BUILDINGS
    # 2: [76, 181, 255],  # MEDIUM DENSITY
    # 3: [97, 105, 255],  # HIGH DENSITY
}


def mask2labels(mask):
    labels = np.zeros(
        (mask.shape[0], mask.shape[1], len(LABEL_COLORS.keys())),
        dtype=np.uint8
    )  # (128, 128, 2)
    for label, color in LABEL_COLORS.items():
        labels[(mask == color).all(axis=-1)] = label

    return labels


def labels2mask(labels):
    mask = np.empty((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for label, color_rgb in LABEL_COLORS.items():
        color_bgr = color_rgb[::-1]
        mask[labels == label] = color_bgr

    return mask


def show_results(img_path, img_mask_path, segmented):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row and 3 columns

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img)
    axes[0].set_title('Input')

    test_img_mask = cv2.imread(img_mask_path, cv2.IMREAD_COLOR)
    test_img_mask = cv2.cvtColor(test_img_mask, cv2.COLOR_BGR2RGB)
    axes[1].imshow(test_img_mask)
    axes[1].set_title('Mask')

    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    axes[2].imshow(segmented)
    axes[2].set_title('Segmented')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    plt.show()


def format_elapsed_time(seconds):
    hours = round(seconds // 3600)
    minutes = round((seconds % 3600) // 60)

    return f"{hours}h {minutes}min"


def get_image_patches(image_path, patch_size=128):
    patches = []

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # (1024, 1920, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image_patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            image_patch = image_patches[i, j, :, :]  # (1, 128, 128, 3)
            image_patch = image_patch.squeeze(0)  # (128, 128, 3)
            patches.append(image_patch)

    return patches
