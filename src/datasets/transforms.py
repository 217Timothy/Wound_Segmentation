import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(512, 512)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        A.Rotate(limit=35, p=0.5),  # 🔥 stronger

        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.2),
            A.ElasticTransform(alpha=1, sigma=50),
        ], p=0.3),

        # 🔥 輕量顏色
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.HueSaturationValue(5, 10, 5, p=0.2),

        A.Resize(img_size[1], img_size[0]),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2(),
    ])


def get_tkr_finetune_train_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),

        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=0,
            p=0.3
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=8,
            val_shift_limit=5,
            p=0.2
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# 驗證集通常只做 Resize (確保尺寸對)，不做翻轉等破壞性增強
def get_val_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])