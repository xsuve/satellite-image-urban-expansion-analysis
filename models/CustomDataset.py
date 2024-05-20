import os
import torch
import numpy as np
from torch.utils.data import Dataset
import utils.helpers as helpers


class CustomDataset(Dataset):
    def __init__(self, data_dir, train_cities):
        self.data_dir = data_dir
        self.train_cities = train_cities

        self.images, self.masks = self.load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # (3, 128, 128)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = self.masks[index]
        mask = np.transpose(mask, (2, 0, 1))  # (2, 128, 128)
        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask)

        return image, mask

    def load_data(self):
        images = []
        masks = []

        for city in self.train_cities:
            for file in city['images']:
                image_path = os.path.join(self.data_dir, city['code'], 'images', file)
                mask_path = os.path.join(self.data_dir, city['code'], 'masks', file)
                image_patches = helpers.get_image_patches(image_path, 128)
                masks_matches = helpers.get_image_patches(mask_path, 128)
                for i in range(len(image_patches)):
                    image = image_patches[i]
                    mask = helpers.mask2labels(masks_matches[i])

                    images.append(image)
                    masks.append(mask)

        print("Images and masks loaded")
        return np.array(images), np.array(masks)
