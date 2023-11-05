import os
import torch

from dataclasses import dataclass


@dataclass
class Config:
    seed = 2024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder_feedforward_dim = 1024
    encoder_nheads = 8
    num_encoder_layer = 2

    vocab_size = 7500
    word_emb_dim = 512
    decoder_hidden_dim = 1024
    num_decoder_layers = 1

    batch = 32
    epoch = 5
    lr = 1e-3

    train_size = 0.8

    # max length for prediction
    max_length = 32

    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'Flick_30k')
    image_dir = os.path.join(dataset_dir, 'Images')
    caption_file = os.path.join(dataset_dir, 'captions.txt')
    vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab' + str(vocab_size) + '.txt')
    encoder_file = ('src/encoder' +
                    '_b' + str(batch) +
                    '_h' + str(decoder_hidden_dim) +
                    '_l' + str(num_decoder_layers) +
                    '_e' + str(epoch) +
                    '.pt')
    decoder_file = ('src/decoder' +
                    '_b' + str(batch) +
                    '_h' + str(decoder_hidden_dim) +
                    '_l' + str(num_decoder_layers) +
                    '_e' + str(epoch) +
                    '.pt')
    embedding_file = ('src/embedding' +
                      '_b' + str(batch) +
                      '_h' + str(decoder_hidden_dim) +
                      '_l' + str(num_decoder_layers) +
                      '_e' + str(epoch) +
                      '.pt')
