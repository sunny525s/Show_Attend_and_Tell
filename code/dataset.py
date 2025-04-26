import json, os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data_path = '/content/data/flickr8k'
batch_size = 32
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CaptionDataset(Dataset):
    def __init__(self, data_path, split, transform, num_captions=5):
        self.img_paths = json.load(open(os.path.join(data_path, f'{split}_img_paths.json'), 'r'))
        self.captions = json.load(open(os.path.join(data_path, f'{split}_captions.json'), 'r'))
        self.caption_lens = json.load(open(os.path.join(data_path, f'{split}_caption_lens.json'), 'r'))
        self.num_captions = num_captions
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
      
        if self.split == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[idx]), torch.tensor(self.caption_lens[idx]).reshape(-1)

        all_captions = [self.captions[i] for i, path in enumerate(self.img_paths) if path == img_path]

        return torch.FloatTensor(img), torch.tensor(self.captions[idx]), torch.tensor(self.caption_lens[idx]).reshape(-1), torch.tensor(all_captions)        

def create_dataset():
    train_dataset = CaptionDataset(data_path, 'train', img_transform)
    val_dataset = CaptionDataset(data_path, 'val', img_transform)
    test_dataset = CaptionDataset(data_path, 'test', img_transform)

    return train_dataset, val_dataset, test_dataset
    
def create_dataloader(batch_size=batch_size):
    train_loader = DataLoader(CaptionDataset(data_path, 'train', img_transform), batch_size=batch_size)
    val_loader = DataLoader(CaptionDataset(data_path, 'val', img_transform), batch_size=batch_size)
    test_loader = DataLoader(CaptionDataset(data_path, 'test', img_transform), batch_size=batch_size)

    return train_loader, val_loader, test_loader
