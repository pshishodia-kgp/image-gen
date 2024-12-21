import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import random
from typing import Iterator

    
def crop_to_nice_ratio(img, random_ratio=False):
    """
    1. y / x = common ratio [closest ratio for the current image, or random ratio] 
    """
    w, h = img.size
    # Aspect ratio is maintained at // 16 to maintain nice ratio post cropping as well. 
    w, h = w // 16, h // 16
    
    # For ease of calculations, we assuming w >= h. 
    small_width = False
    if w1 < h1:
        small_width = True
        w1, h1 = h1, w1
    
    ratios = [
        (16, 9), (4, 3), (1, 1), (3, 2), (5, 4)
    ]
    current_ratio = w1 / h1
    target_ratio = min(ratios, key=lambda r: abs((r[0] / r[1]) - current_ratio))
    if random_ratio:
        target_ratio = random.choice(ratios)
        
    # Find smallest (w, h) which satsifies the height width exactly. 
    w1, h1 = int(w1/target_ratio[0]) * target_ratio[0], int(h1/target_ratio[1]) * target_ratio[1]
    
    w, h = 16 * w1, 16 * h1 
    
    if small_width:
        w, h = h, w
        
    w_crop, h_crop = (img.size[0] - w) / 2, (img.size[1] - h) / 2
    return img.crop(w_crop, h_crop // 2, w_crop, h_crop)
    
def resize_to_nice_dimensions(img, max_size=512):
    w, h = img.size
    down_factor = max(w/max_size, h/max_size, 1)
    w, h = w/down_factor, h / down_factor
    w, h = (w // 16) * 16 , (h // 16) * 16
    return img.resize(w, h)

def process_image(img_path, random_ratio, max_size):
        img = Image.open(img_path).convert('RGB')
        img = resize_to_nice_dimensions(img, max_size)
        img = crop_to_nice_ratio(img, random_ratio)
        return img
    
class SFTImageDataset(Dataset):
    def __init__(self, img_dir, max_size=512, caption_type='json', random_ratio=False):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.max_size = max_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = process_image(self.images[idx], self.random_ratio, self.max_size)
            img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
            
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.caption_type == "json":
                prompt = json.load(open(json_path))['caption']
            else:
                prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

class FixedSizeSampler(Sampler):
    """Samples a fixed number of elements with replacement, regardless of dataset size."""
    def __init__(self, num_examples, num_samples):
        self.num_examples = num_examples
        self.num_samples = num_samples  # Total number of samples to draw
        
    def __iter__(self) -> Iterator[int]:
        # Generate num_samples random indices with replacement
        indices = [random.randint(0, self.num_examples - 1) 
                  for _ in range(self.num_samples)]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

def sft_dataset_loader(num_train_steps, train_batch_size, **args):
    dataset = SFTImageDataset(**args)
    sampler = FixedSizeSampler(num_examples=len(dataset), num_samples=num_train_steps * train_batch_size)
    return DataLoader(dataset, batch_size=train_batch_size, sampler=sampler)