from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms as T
import os
import glob

def get_paths_from_images(dataroot):
    """Get all image paths from a directory"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(dataroot, '**', ext), recursive=True))
        paths.extend(glob.glob(os.path.join(dataroot, ext)))
    return sorted(paths)

def transform_augment(imgs, split='train'):
    """Apply data augmentation transforms"""
    if split == 'train':
        # Random horizontal flip
        if random.random() < 0.5:
            imgs = [torch.flip(img, [2]) for img in imgs]
        
        # Random vertical flip
        if random.random() < 0.5:
            imgs = [torch.flip(img, [1]) for img in imgs]
        
        # Random rotation (90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            imgs = [torch.rot90(img, k, [1, 2]) for img in imgs]
    
    return imgs

class LRHRRefDataset(Dataset):
    def __init__(
        self,
        dataroot_hr,
        dataroot_lr,
        dataroot_ref,
        split="train",
        data_len=-1,
        patch_size=None,
        use_ColorJitter=False,
        use_gray=False,
        gt_as_ref=False
    ):
        self.data_len = data_len
        self.split = split
        self.patch_size = patch_size
        self.use_ColorJitter = use_ColorJitter
        self.gt_as_ref = gt_as_ref
        self.use_gray = use_gray

        self.lr_path = get_paths_from_images(dataroot_lr)
        self.hr_path = get_paths_from_images(dataroot_hr)
        self.ref_path = get_paths_from_images(dataroot_ref)

        # Ensure all datasets have the same length
        min_len = min(len(self.lr_path), len(self.hr_path), len(self.ref_path))
        self.lr_path = self.lr_path[:min_len]
        self.hr_path = self.hr_path[:min_len]
        self.ref_path = self.ref_path[:min_len]

        self.jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ) if use_ColorJitter else None

        dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = dataset_len
        else:
            self.data_len = min(self.data_len, dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Load images
        img_HR = (
            np.asarray(Image.open(self.hr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_Ref = (
            np.asarray(Image.open(self.ref_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_LR = (
            np.asarray(Image.open(self.lr_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)

        # Convert to torch tensors (HWC -> CHW)
        img_HR = torch.from_numpy(img_HR).permute(2, 0, 1)
        img_Ref = torch.from_numpy(img_Ref).permute(2, 0, 1)
        img_LR = torch.from_numpy(img_LR).permute(2, 0, 1)

        if self.gt_as_ref and random.random() < 0.3:
            img_Ref = img_HR.clone()

        # Data augmentation
        [img_LR, img_Ref, img_HR] = transform_augment(
            [img_LR, img_Ref, img_HR], split=self.split
        )

        # Apply color jitter or grayscale conversion to reference image
        if self.use_gray and random.random() < 0.3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img_Ref.dtype, device=img_Ref.device)
            gray = (img_Ref * weights[:, None, None]).sum(dim=0)  # shape: (H, W)
            img_Ref = gray.unsqueeze(0).repeat(3, 1, 1)
        elif self.jitter is not None:
            img_Ref_np = img_Ref.permute(1, 2, 0).numpy()
            img_Ref_PIL = Image.fromarray((img_Ref_np * 255).astype(np.uint8))
            img_Ref_PIL = self.jitter(img_Ref_PIL)
            img_Ref = torch.from_numpy(np.asarray(img_Ref_PIL).astype(np.float32) / 255.0)
            img_Ref = img_Ref.permute(2, 0, 1)

        # Random cropping
        if self.patch_size is not None:
            _, h, w = img_HR.shape
            ps = self.patch_size

            if h < ps or w < ps:
                # Pad if image is smaller than patch size
                pad_h = max(0, ps - h)
                pad_w = max(0, ps - w)
                if pad_h > 0 or pad_w > 0:
                    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
                    img_HR = torch.nn.functional.pad(img_HR, padding, mode='reflect')
                    img_LR = torch.nn.functional.pad(img_LR, padding, mode='reflect')
                    img_Ref = torch.nn.functional.pad(img_Ref, padding, mode='reflect')
                    _, h, w = img_HR.shape

            rnd_h = random.randint(0, h - ps) if h > ps else 0
            rnd_w = random.randint(0, w - ps) if w > ps else 0

            img_HR = img_HR[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]
            img_LR = img_LR[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]
            img_Ref = img_Ref[:, rnd_h : rnd_h + ps, rnd_w : rnd_w + ps]

        return {
            "HR": 2 * img_HR - 1,  # Normalize to [-1, 1]
            "LR": 2 * img_LR - 1,
            "Ref": 2 * img_Ref - 1,
            "path": self.hr_path[index],
            "txt": "",
        }