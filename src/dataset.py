import json, os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CaptionDataset(Dataset):
    def __init__(self, data_path, split, transform, num_captions=5):
        self.img_paths = json.load(
            open(os.path.join(data_path, f"{split}_img_paths.json"), "r")
        )
        self.captions = json.load(
            open(os.path.join(data_path, f"{split}_captions.json"), "r")
        )
        self.caption_lens = json.load(
            open(os.path.join(data_path, f"{split}_caption_lens.json"), "r")
        )
        self.num_captions = num_captions
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.split == "train":
            return (
                torch.FloatTensor(img),
                torch.tensor(self.captions[idx]),
                torch.tensor(self.caption_lens[idx]).reshape(-1),
            )

        all_captions = [
            self.captions[i]
            for i, path in enumerate(self.img_paths)
            if path == img_path
        ]

        return (
            torch.FloatTensor(img),
            torch.tensor(self.captions[idx]),
            torch.tensor(self.caption_lens[idx]).reshape(-1),
            torch.tensor(all_captions),
        )


def create_dataloaders(transform, caption_path="captions", batch_size=32, workers=1):
    """
    Creates DataLoaders for train, validation, and test splits.

    Args:
        transform (callable): The image transformations to apply.
        caption_path (str): Path to caption JSON files.
        batch_size (int): Batch size for DataLoaders.
        workers (int): Number of workers for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = CaptionDataset(caption_path, "train", transform)
    val_dataset = CaptionDataset(caption_path, "val", transform)
    test_dataset = CaptionDataset(caption_path, "test", transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    return train_loader, val_loader, test_loader
