import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'


LABEL_COLORS = {
    0: [126, 212, 140],  # LAND
    1: [109, 214, 248],  # BUILDINGS
    # 2: [76, 181, 255],  # MEDIUM DENSITY
    # 3: [97, 105, 255],  # HIGH DENSITY
}


def process_img(image_path, ismask):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    if not ismask:
        image = image.transpose((2, 0, 1))  # Convert from HWC to CHW format
        image = torch.from_numpy(image).float()
    else:
        label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        for label, color in LABEL_COLORS.items():
            label_mask[(image == color).all(axis=2)] = label
        image = torch.from_numpy(label_mask).long()

    return image


def show_results(test_img_path, test_img_mask_path, segmented):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row and 3 columns

    test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(test_img)
    axes[0].set_title('Input')

    test_img_mask = cv2.imread(test_img_mask_path, cv2.IMREAD_COLOR)
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


def rename_files(from_dir, to_dir):
    for _, filename in enumerate(os.listdir(from_dir)):
        if filename.startswith("img_"):
            file_index = int(filename[filename.find('_')+len('_'):filename.rfind('.')])

            if file_index < 10:
                os.rename(from_dir + filename, to_dir + '000' + str(file_index) + '.jpg')
            elif 10 <= file_index < 100:
                os.rename(from_dir + filename, to_dir + '00' + str(file_index) + '.jpg')
            elif 100 <= file_index < 1000:
                os.rename(from_dir + filename, to_dir + '0' + str(file_index) + '.jpg')
            else:
                os.rename(from_dir + filename, to_dir + str(file_index) + '.jpg')
