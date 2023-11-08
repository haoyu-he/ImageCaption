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
    num_gpt1_layers = 6
    n_head = 8

    batch = 32
    epoch = 10
    lr_lstm = 5e-4
    lr_gpt1 = 2e-4

    train_size = 0.8

    max_length = 128

    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'Flick_30k')
    image_dir = os.path.join(dataset_dir, 'Images')
    caption_file = os.path.join(dataset_dir, 'captions.txt')
    vocab_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab' + str(vocab_size) + '.txt')

    @property
    def encoder_lstm_file(self) -> str:
        return (
                'src/encoder' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_lstm_layers) +
                '_e' + str(self.epoch) +
                '_lstm.pt'
        )

    @property
    def decoder_lstm_file(self) -> str:
        return (
                'src/decoder' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_lstm_layers) +
                '_e' + str(self.epoch) +
                '_lstm.pt'
        )

    @property
    def embedding_lstm_file(self) -> str:
        return (
                'src/embedding' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_lstm_layers) +
                '_e' + str(self.epoch) +
                '_lstm.pt'
        )

    @property
    def encoder_gpt1_file(self) -> str:
        return (
                'src/encoder' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_gpt1_layers) +
                '_nh' + str(self.n_head) +
                '_e' + str(self.epoch) +
                '_gpt1.pt'
        )

    @property
    def decoder_gpt1_file(self) -> str:
        return (
                'src/decoder' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_gpt1_layers) +
                '_nh' + str(self.n_head) +
                '_e' + str(self.epoch) +
                '_gpt1.pt'
        )

    @property
    def embedding_gpt1_file(self) -> str:
        return (
                'src/embedding' +
                '_b' + str(self.batch) +
                '_h' + str(self.hidden_dim) +
                '_l' + str(self.num_gpt1_layers) +
                '_nh' + str(self.n_head) +
                '_e' + str(self.epoch) +
                '_gpt1.pt'
        )
