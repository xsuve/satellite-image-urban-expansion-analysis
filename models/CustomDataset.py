import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.helpers as helpers


class CustomDataset(Dataset):
    def __init__(self, data_dir, train_cities):
        self.data_dir = data_dir
        self.train_cities = train_cities

        self.images, self.masks = self.load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        return image, mask

    def load_data(self):
        images = []
        masks = []

        for city in self.train_cities.keys():
            for cp in self.train_cities[city]:
                for file in os.listdir(os.path.join(self.data_dir, city, cp, 'images')):
                    if file.endswith('.jpg'):
                        # Load image
                        image_path = os.path.join(self.data_dir, city, cp, 'images', file)
                        image = helpers.process_img(image_path, ismask=False)

                        # Load corresponding mask
                        mask_path = os.path.join(self.data_dir, city, cp, 'masks', file)
                        mask = helpers.process_img(mask_path, ismask=True)

                        # Append image and mask
                        images.append(image)
                        masks.append(mask)

        print("Images and masks loaded")
        return np.array(images), np.array(masks)
