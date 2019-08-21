#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from albumentations import (Compose, RandomRotate90, HorizontalFlip,
    VerticalFlip, RandomBrightness, RandomContrast, ShiftScaleRotate,
)


#------------------------------------------------------------------------------
#  Augmenter
#------------------------------------------------------------------------------
class Augmentation(object):
    def __init__(self, p=0.5, rot90=0.0, flip_hor=0.5, flip_ver=0.5,
                brightness=0.2, contrast=0.1, shift=0.1625, scale=0.6, rotate=10):
        
        self.augmenter = Compose([
            RandomRotate90(p=rot90),
            HorizontalFlip(p=flip_hor),
            VerticalFlip(p=flip_ver),
            RandomBrightness(p=brightness, limit=0.2),
            RandomContrast(p=contrast, limit=0.2),
            ShiftScaleRotate(shift_limit=shift, scale_limit=scale, rotate_limit=rotate, p=0.7),
        ], p=p)

    def __call__(self, image):
        data = self.augmenter(image=image)
        return data['image']
