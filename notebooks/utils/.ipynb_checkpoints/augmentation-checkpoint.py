import albumentations as A
from fastai.vision import *
# from fastbook import *
from fastai.vision.all import *
from fastai.vision.widgets import *
import cv2
import numpy as np


class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx,order=None,2
    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
    

# def get_train_aug(RESOLUTION=380): 
#     return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                          always_apply=True),
#         A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.2, 1), \
#                             interpolation=cv2.INTER_CUBIC),
#         A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
#         A.FancyPCA(p=0.8, alpha=0.5),
#         A.Transpose(p=0.7),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.4),
#         A.ShiftScaleRotate(p=0.7),
#         A.Rotate(p=0.8),
#         A.HueSaturationValue(
#             hue_shift_limit=0.3, 
#             sat_shift_limit=0.3, 
#             val_shift_limit=0.3, 
#             p=0.7
#         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=(-0.4,0.4), 
#             contrast_limit=(-0.4, 0.4), 
#             p=0.7
#         ),
#         A.CoarseDropout(p=0.8, max_holes=30),
#         A.Cutout(p=0.8, max_h_size=40, max_w_size=40),
#         A.OneOf([
#                 A.OpticalDistortion(p=0.3),
#                 A.GridDistortion(p=.1),
#                 A.IAAPiecewiseAffine(p=0.3),
#                 ], p=0.6),
#         A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
#         A.OneOf([
#             A.IAAAdditiveGaussianNoise(p=1.0),
#             A.GaussNoise(p=1.0),
#             A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 3))
#             ], p=0.5),
#         ], p=1.0)



# def get_valid_aug(RESOLUTION=380): 
#     return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                      always_apply=True),
#         A.OneOf([
#             A.CenterCrop(RESOLUTION,RESOLUTION, always_apply=True),
#             A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.4, 1.0), \
#                                 always_apply=True, interpolation=cv2.INTER_CUBIC),
#             ], p=1.0),
#         A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  
#         A.HorizontalFlip(p=0.5),
#         A.FancyPCA(p=0.75, alpha=0.5),
#     #     A.HueSaturationValue(
#     #         hue_shift_limit=0.2, 
#     #         sat_shift_limit=0.2, 
#     #         val_shift_limit=0.2, 
#     #         p=0.6
#     #         ),
#     #     A.RandomBrightnessContrast(
#     #         brightness_limit=(-0.1,0.1), 
#     #         contrast_limit=(-0.1, 0.1), 
#     #         p=0.6
#     #         ),
#         A.Sharpen(p=1.0, alpha=(0.1, 0.3), lightness=(0.3, 0.9))
#         ], p=1.0)


# limit to normal values of "greenish" as sampled from the images
def get_rand_dropout_color() :
    rng = np.random.default_rng(12345)
    
    g = rng.integers(low=85, high=225)
    r = rng.integers(low=55, high=int(0.7*g))
    b = rng.integers(low=35, high=int(0.6*(g-40)))
    return [r,g,b]

    
    
def get_train_aug(RESOLUTION=380): 
    
    
    augs = A.Compose([
        A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION*2, min_width=RESOLUTION*2, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.4, 1), \
                            interpolation=cv2.INTER_CUBIC),
        A.OneOf([
            A.CenterCrop(RESOLUTION,RESOLUTION, p=0.8),
            A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.3, 1.0), \
                                p=0.8, interpolation=cv2.INTER_CUBIC),
            ], p=1.0),
        A.ShiftScaleRotate(p=0.4),       
        A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION, min_width=RESOLUTION, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
        A.FancyPCA(p=0.8, alpha=0.5),
#         A.Transpose(p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
#         A.HueSaturationValue(
#             always_apply=False, p=0.3, 
#             hue_shift_limit=(-20, 20), 
#             sat_shift_limit=(-30, 30), 
#             val_shift_limit=(-20, 20)),

        A.HueSaturationValue(
            hue_shift_limit=0.3, #.3
            sat_shift_limit=0.3, #.3
            val_shift_limit=0.2, #.3
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3,0.2), #-.2,.2
            contrast_limit=(-0.3, 0.2),  #-.2,.2
            #p=0.6
        ),
        A.CoarseDropout(p=0.8, max_holes=30, min_height=int(RESOLUTION/180), max_height=int(RESOLUTION/90), 
                        min_width=int(RESOLUTION/180), max_width=int(RESOLUTION/90), 
                        fill_value=[int(i/3) for i in get_rand_dropout_color()]),
#         A.Cutout(p=0.8, max_h_size=40, max_w_size=40),
        A.Cutout(p=1, max_h_size=int(RESOLUTION/12), max_w_size=int(RESOLUTION/20), num_holes=16, fill_value=get_rand_dropout_color()),
        A.Cutout(p=1, max_h_size=int(RESOLUTION/20), max_w_size=int(RESOLUTION/12), num_holes=16, fill_value=get_rand_dropout_color()),
        A.OneOf([
                A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.6599999666213989, 0.6800000071525574), 
                                    shift_limit=(-0.6699999570846558, 0.4599999785423279), interpolation=0, 
                                    border_mode=0, value=(0, 0, 0), mask_value=None),
#                 A.OpticalDistortion(p=0.5, distort_limit=0.15, shift_limit=0.15),
#                 A.GridDistortion(p=0.5, distort_limit=0.5),
                A.GridDistortion(always_apply=False, p=1.0, 
                                 num_steps=6, distort_limit=(-0.4599999785423279, 0.5), 
                                 interpolation=0, border_mode=0, 
                                 value=(0, 0, 0), mask_value=None),

#                 A.IAAPiecewiseAffine(p=0.5, scale=(0.1, 0.14)),
                ], p=0.6),
        A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
#         A.GaussNoise(var_limit=(300.0, 500.0), p=0.4),


        A.OneOf([
            A.Equalize(always_apply=False, p=0.9, mode='cv', by_channels=True),
#             A.Solarize(always_apply=False, p=0.9, threshold=(67, 120)),
#             A.IAAAdditiveGaussianNoise(p=1.0),
            A.GaussNoise(p=0.9),
            A.MotionBlur(always_apply=False, p=0.9, blur_limit=(4, 10)),
            A.ISONoise(always_apply=False, p=0.9, 
               intensity=(0.10000000149011612, 1.399999976158142), 
               color_shift=(0.009999999776482582, 0.4000000059604645)),
            ], p=0.5),
        ], p=1.0)

    return augs


def get_valid_aug(RESOLUTION=380): 
    return A.Compose([
        A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
                     always_apply=True),
        A.OneOf([
            A.CenterCrop(RESOLUTION,RESOLUTION, p=0.8),
            A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.5, 1.0), \
                                p=0.8, interpolation=cv2.INTER_CUBIC),
            ], p=1.0),
        A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION, min_width=RESOLUTION, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  # just in case
        A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.2),
        A.FancyPCA(p=1.0, alpha=0.5),
#         A.HueSaturationValue(
#             hue_shift_limit=0.1, 
#             sat_shift_limit=0.1, 
#             val_shift_limit=0.1, 
#             p=0.5
#             ),
#         A.RandomBrightnessContrast(
#             brightness_limit=(-0.1,0.1), 
#             contrast_limit=(-0.1, 0.1), 
#             p=0.5
#             ),
        A.Sharpen(p=1.0, alpha=(0.1, 0.3), lightness=(0.3, 0.9))
        ], p=1.0)



def get_tta_aug(RESOLUTION=380): 
    return A.Compose([
        A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
                     always_apply=True),
        A.OneOf([
            A.CenterCrop(RESOLUTION,RESOLUTION, p=0.8),
            A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.5, 1.0), \
                                p=0.8, interpolation=cv2.INTER_CUBIC),
            ], p=1.0),
        A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION, min_width=RESOLUTION, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  # just in case
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.FancyPCA(p=1.0, alpha=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.1, 
            sat_shift_limit=0.1, 
            val_shift_limit=0.1, 
            p=0.5
            ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1,0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5
            ),
        A.Sharpen(p=1.0, alpha=(0.1, 0.3), lightness=(0.3, 0.9))
        ], p=1.0)