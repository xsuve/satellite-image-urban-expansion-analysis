import cv2
import torch
import numpy as np
from models.UNet.UNet import UNet
import utils.helpers as helpers
import utils.model as model
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

'''
    Google Earth Pro:
        Resolution: 1920x1080 -> Crop to 1920x1024
        Scale: 1200m
'''

if __name__ == '__main__':
    train_cities = [
        {
            'code': 'SB',
            'images': [
                'E_09-2021.jpg',
                'N_09-2019.jpg',
                'W_09-2016.jpg'
            ]
        }
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(helpers.LABEL_COLORS.keys())
    unet = UNet(in_c=3, num_classes=num_classes).to(device)

    phase = 'train'
    # phase = 'segment'
    # phase = 'analyse'
    BATCH_SIZE = 1
    NUM_EPOCH = 2

    if phase == 'train':
        model.train(
            unet,
            'data/',
            train_cities,
            'output/',
            device,
            BATCH_SIZE,
            NUM_EPOCH
        )
    elif phase == 'segment':
        img_path = 'data/SB/images/S_10-2022.jpg'
        img_mask_path = 'data/SB/masks/S_10-2022.jpg'

        segmented = model.segment(
            unet,
            'output/07-06-2024_11-53-18_1-3.pth',
            device,
            img_path
        )

        helpers.show_results(
            img_path,
            img_mask_path,
            segmented
        )
    else:
        timelapse = [
            {'year': 2011, 'image': 'S_05-2011.jpg'},
            # {'year': 2013, 'image': 'S_04-2013.jpg'},
            # {'year': 2015, 'image': 'S_06-2015.jpg'},
            # {'year': 2018, 'image': 'S_04-2018.jpg'},
            # {'year': 2021, 'image': 'S_02-2021.jpg'},
            {'year': 2022, 'image': 'S_10-2022.jpg'}
        ]

        for item in timelapse:
            img_path = 'data/SB/images/' + item['image']

            segmented = model.segment(
                unet,
                'output/23-05-2024_23-09-28_1-3.pth',
                device,
                img_path
            )
            segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            item['image'] = image
            item['segmented'] = segmented

        ##
        fig, axes = plt.subplots(2, len(timelapse), figsize=(15, 5))
        for i, item in enumerate(timelapse):
            axes[0, i].imshow(item['image'])
            axes[0, i].set_title(item['year'])

            axes[1, i].imshow(item['segmented'])

            axes[0, i].axis('off')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

        ##
        height, width, _ = timelapse[0]['segmented'].shape
        heatmap = np.zeros((height, width), dtype=np.float32)

        prev = cv2.cvtColor(timelapse[0]['segmented'], cv2.COLOR_RGB2GRAY)
        for item in timelapse[1:]:
            current = cv2.cvtColor(item['segmented'], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(prev, current)
            heatmap += diff.astype(np.float32)
            prev = current

        num_frames = len(timelapse) - 1
        filtered_diff = heatmap / num_frames
        heatmap_normalized = cv2.normalize(filtered_diff, None, 0, 255, cv2.NORM_MINMAX)
        # percentile = np.percentile(heatmap, 99)
        # heatmap_normalized = np.clip(heatmap, 0, percentile)
        # heatmap_normalized = (heatmap_normalized / percentile) * 255
        # heatmap_normalized = heatmap_normalized.astype(np.uint8)

        heatmap_blurred = cv2.GaussianBlur(heatmap_normalized, (151, 151), 0)
        heatmap_blurred = heatmap_blurred.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

        opacity = 0.4
        image = timelapse[len(timelapse) - 1]['image'].copy()
        overlayed = cv2.addWeighted(image, 1 - opacity, heatmap_colored, opacity, 0)

        cv2.imshow('Analysis', overlayed)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
