import torch
from models.UNet.UNet import UNet
import utils.helpers as helpers
import utils.model as model

'''
    Google Earth Pro:
        - 1280x720 (720 HD)
        - Free view scale: 9.96 km
        - Image scale: 10 km
    
    Adobe Illustrator:
        - Crop: 512x512
    
    Cities:
        001 - SB
        002 - AB
        003 - CJ
        004 - BV
        005 - IS
        006 - DJ
        007 - TM
        008 - PH
        009 - MS
        010 - B
        011 - BR
        012 - GL
        013 - TL
        014 - BZ
        015 - VL
        016 - OT
        017 - MH
        018 - AR
        019 - BH
        020 - BN
        021 - HD
'''

'''
    07/2022
    1920x1080
    1052m
'''

if __name__ == '__main__':
    # helpers.rename_files('data/SB/S/m/', 'data/SB/S/masks/')

    train_cities = {
        'SB': ['S']
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(helpers.LABEL_COLORS.keys())
    unet = UNet(num_classes=num_classes).to(device)

    phase = 'segment'  # train | segment

    if phase == 'train':
        model.train(
            unet,
            'data/',
            train_cities,
            'output/',
            device,
            1,
            15
        )
    else:
        test_img_path = 'data/SB/S/S.jpg'
        test_img_mask_path = 'data/SB/S/S-mask.jpg'

        segmented = model.segment(
            unet,
            'output/21-04-2024_19-29-16.pth',
            test_img_path
        )

        helpers.show_results(
            test_img_path,
            test_img_mask_path,
            segmented
        )
