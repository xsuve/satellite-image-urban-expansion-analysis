import torch
from models.UNet.UNet import UNet
import utils.helpers as helpers
import utils.model as model

'''
    Google Earth Pro:
        07/2022
        1920x1080 -> Crop to 1920x1024
        1052m
'''

if __name__ == '__main__':
    train_cities = [
        {
            'code': 'SB',
            'images': [
                'S_07-2022.jpg',
                # 'N_07-2022.jpg'
            ]
        }
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(helpers.LABEL_COLORS.keys())
    unet = UNet(in_c=3, num_classes=num_classes).to(device)

    phase = 'segment'  # train | segment

    if phase == 'train':
        model.train(
            unet,
            'data/',
            train_cities,
            'output/',
            device,
            2,
            2
        )
    else:
        img_path = 'data/SB/images/N_07-2022.jpg'
        img_mask_path = 'data/SB/images/N_07-2022.jpg'

        segmented = model.segment(
            unet,
            'output/20-05-2024_14-09-24.pth',
            device,
            img_path
        )

        helpers.show_results(
            img_path,
            img_mask_path,
            segmented
        )
