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
    num_lstm_layers = 1
    num_transformer_layers = 6
    n_head = 8

    batch = 32
    epoch = 5
    lr_lstm = 2e-3
    lr_transformer = 2e-4

    train_size = 0.8

    max_length = 128

    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'Flick_30k')
    image_dir = os.path.join(dataset_dir, 'Images')
    caption_file = os.path.join(dataset_dir, 'captions.txt')
    vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab' + str(vocab_size) + '.txt')
    encoder_lstm_file = (
            'src/encoder' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_lstm.pt'
    )
    decoder_lstm_file = (
            'src/decoder' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_lstm.pt'
    )
    embedding_lstm_file = (
            'src/embedding' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_lstm.pt'
    )
    encoder_transformer_file = (
            'src/encoder' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_transformer.pt'
    )
    decoder_transformer_file = (
            'src/decoder' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_transformer.pt'
    )
    embedding_transformer_file = (
            'src/embedding' +
            '_b' + str(batch) +
            '_h' + str(hidden_dim) +
            '_l' + str(num_lstm_layers) +
            '_e' + str(epoch) +
            '_transformer.pt'
    )
