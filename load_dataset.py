import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image

from vocab import Vocab


def preprocess_image():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class Flicker30k:

    def __init__(self,
                 image_folder: str,
                 caption_file: str,
                 vocab: Vocab,
                 transform=lambda x: x):

        self.image_folder = image_folder
        self.transform = transform

        self.samples = []

        with open(caption_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):

                # ignore header line
                if i == 0:
                    continue

                instance = line.strip().lower().split(",", 1)
                image_id = instance[0]
                caption = vocab.sos + instance[1] + vocab.eos

                # split words
                words = vocab.word_tokenize(caption)

                word_ids = [vocab.get_index(word) for word in words]

                sample = {
                    'image_id': image_id,
                    'words_ids': torch.tensor(word_ids, dtype=torch.long)
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_file = os.path.join(self.image_folder, sample['image_id'])

        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)

        return image, sample['words_ids']

