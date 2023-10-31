import os

import torch
from torch.utils.data import random_split

from config import Config
from vocab import Vocab
from load_dataset import Flicker30k, preprocess_image

config = Config
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

# build vocabulary
vocab = Vocab()
vocab.load_vocab(config.vocab_file)

# load dataset
dataset = Flicker30k(config.dataset_dir, config.caption_file, vocab, preprocess_image)
train_size = int(config.train_size * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
pass