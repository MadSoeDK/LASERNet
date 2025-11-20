import glob
import torch 
from pathlib import Path
from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

#00-09 train
#09-14: val
#14-20: test

class Temperature(Dataset):
    """
    Loads temperature image sequences for next-frame prediction:
    seq_length frames → 1 target frame.
    """

    def __init__(self, root_dir, folders, seq_length=5, transform=None):
        self.seq_length = seq_length
        self.transform = transform

        all_images = []
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            paths = sorted(glob.glob(os.path.join(folder_path, "*_t_1.tiff")))
            all_images.extend(paths)

        print(f"[DEBUG] Loaded {len(all_images)} temperature frames from {folders}")

        if len(all_images) < seq_length + 1:
            print("⚠ WARNING: Not enough frames to build sequences")

        # Build (sequence → target) pairs across ALL frames
        self.samples = []
        for i in range(len(all_images) - seq_length):
            seq_paths = all_images[i : i + seq_length]
            target_path = all_images[i + seq_length]
            self.samples.append((seq_paths, target_path))

        print(f"[DEBUG] Built {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_paths, target_path = self.samples[idx]

        seq_frames = []
        for p in seq_paths:
            img = Image.open(p).convert("L")
            img = img.resize((512, 256))
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
            seq_frames.append(tensor)

        seq_tensor = torch.stack(seq_frames, dim=0)  # [seq_len, 1, H, W]

        target_img = Image.open(target_path).convert("L")
        target_img = target_img.resize(512, 256)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)

        return seq_tensor, target_tensor



def get_image_dataloaders(root_dir, seq_length=5, batch_size=2):
    train_folders = [f"{i:02d}" for i in range(0, 10)]
    val_folders   = [f"{i:02d}" for i in range(10, 16)]
    test_folders  = [f"{i:02d}" for i in range(16, 25)]

    train_ds = Temperature(root_dir, train_folders, seq_length)
    val_ds   = Temperature(root_dir, val_folders, seq_length)
    test_ds  = Temperature(root_dir, test_folders, seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# TEST
#path_to_file = "/Users/eva/Documents/DataSpatiotemporal/"

#train_loader, val_loader, test_loader = get_image_dataloaders(root_dir=path_to_file,seq_length=5, batch_size=2,)
"""
for seq, target in train_loader:
    print(seq.shape, target.shape)
    break
"""