import os
import torch

from dataclasses import dataclass


@dataclass
class Config:
    seed = 2024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab_size = 7500
    word_emb_dim = 512
    hidden_dim = 1024
    num_layers = 1

    batch = 32
    epoch = 5
    lr = 1e-3

    train_size = 0.8

    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'Flick_30k')
    image_dir = os.path.join(dataset_dir, 'Images')
    caption_file = os.path.join(dataset_dir, 'captions.txt')
    vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab' + str(vocab_size) + '.txt')
    encoder_file = ('src/encoder' +
                    '_b' + str(batch) +
                    '_h' + str(hidden_dim) +
                    '_l' + str(num_layers) +
                    '_e' + str(epoch) +
                    '.pt')
    decoder_file = ('src/decoder' +
                    '_b' + str(batch) +
                    '_h' + str(hidden_dim) +
                    '_l' + str(num_layers) +
                    '_e' + str(epoch) +
                    '.pt')
    embedding_file = ('src/embedding' +
                      '_b' + str(batch) +
                      '_h' + str(hidden_dim) +
                      '_l' + str(num_layers) +
                      '_e' + str(epoch) +
                      '.pt')
