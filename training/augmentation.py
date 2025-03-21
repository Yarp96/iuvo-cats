import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(256, 256)):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.5, 1.0), p=1.0),
        #A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, scale_limit=0.05, rotate_limit=20, border_mode=0),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_test_transforms(img_size=(256, 256)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(p=1.0),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))