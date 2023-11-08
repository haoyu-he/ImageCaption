import os
import argparse

import torch
from torch.utils.data import random_split, DataLoader

from tqdm import tqdm

from config import Config
from vocab import Vocab
from load_dataset import Flicker30k, preprocess_image, Padding
from model import Encoder, DecoderLSTM, DecoderGPT1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gpt1'])
args = parser.parse_args()

config = Config()
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

# load vocabulary
vocab = Vocab()
vocab.load_vocab(config.vocab_file)

# load dataset
print('---Loading dataset---')

dataset = Flicker30k(config.image_dir, config.caption_file, vocab, preprocess_image())
train_size = int(config.train_size * len(dataset))
val_size = len(dataset) - train_size
collate_fn = Padding(pad_index=vocab.word2index[vocab.pad])
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch, shuffle=False, collate_fn=collate_fn)

# create model
print('---Initializing model---')

encoder = Encoder(word_emb_dim=config.word_emb_dim).to(config.device)
emb_layer = torch.nn.Embedding(num_embeddings=config.vocab_size,
                               embedding_dim=config.word_emb_dim,
                               padding_idx=vocab.word2index[vocab.pad]).to(config.device)
if args.model == 'lstm':
    decoder = DecoderLSTM(word_emb_dim=config.word_emb_dim,
                          hidden_dim=config.hidden_dim,
                          num_layers=config.num_lstm_layers,
                          vocab_size=config.vocab_size).to(config.device)
else:
    decoder = DecoderGPT1(word_emb_dim=config.word_emb_dim,
                          nhead=config.n_head,
                          hidden_dim=config.hidden_dim,
                          num_layers=config.num_gpt1_layers,
                          vocab_size=config.vocab_size).to(config.device)

criterion = torch.nn.CrossEntropyLoss().to(config.device)
parameters = list(encoder.parameters()) + list(emb_layer.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params=parameters, lr=config.lr_lstm if args.model == 'lstm' else config.lr_gpt1)

# training
print('---Training---')

if not os.path.exists('./src'):
    os.mkdir('./src')

for epoch in range(config.epoch):

    print('# train epoch', epoch)
    for i, batch in enumerate(tqdm(train_loader)):

        encoder.train()
        emb_layer.train()
        decoder.train()

        image_batch, caption_batch = batch[0].to(config.device), batch[1].to(config.device)

        caption_emb = emb_layer(caption_batch).permute(1, 0, 2)
        # caption_emb: (caption_length, batch, word_emb_dim)
        seq_length = caption_emb.shape[0]
        batch_size = caption_emb.shape[1]

        # get image embedding
        image_emb = encoder(image_batch).unsqueeze(0)
        # image_emb: (1, batch, word_emb_dim)

        # feed decoder
        decoder_input = torch.cat([image_emb, caption_emb], dim=0)
        if args.model == 'lstm':
            hidden = decoder.hidden_0.repeat(1, batch_size, 1)
            cell = decoder.cell_0.repeat(1, batch_size, 1)
            # (num_layers, batch, hidden_dim)

            # prepare output and target
            output, _ = decoder(decoder_input, hidden, cell)
            # output: (caption_length + 1, batch, vocab_size)
        else:
            output = decoder(decoder_input)
            # output: (caption_length + 1, batch, vocab_size)
        output = output[1:-1, :, :].view(-1, config.vocab_size)

        targets = caption_batch.permute(1, 0)[1:, :].reshape(-1)
        mask = targets != vocab.word2index[vocab.pad]
        # only compare non-pad tokens
        output = output[mask, :]
        targets = targets[mask]

        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()

    print('# evaluate')
    acc = []
    for i, batch in enumerate(tqdm(val_loader)):

        encoder.eval()
        emb_layer.eval()
        decoder.eval()

        with torch.no_grad():

            image_batch, caption_batch = batch[0].to(config.device), batch[1].to(config.device)

            caption_emb = emb_layer(caption_batch).permute(1, 0, 2)
            # caption_emb: (caption_length, batch, word_emb_dim)
            seq_length = caption_emb.shape[0]
            batch_size = caption_emb.shape[1]

            # get image embedding
            image_emb = encoder(image_batch).unsqueeze(0)
            # image_emb: (1, batch, word_emb_dim)

            # feed decoder
            decoder_input = torch.cat([image_emb, caption_emb], dim=0)
            if args.model == 'lstm':
                hidden = decoder.hidden_0.repeat(1, batch_size, 1)
                cell = decoder.cell_0.repeat(1, batch_size, 1)
                # (num_layers, batch, hidden_dim)

                # prepare output and target
                output, _ = decoder(decoder_input, hidden, cell)
                # output: (caption_length + 1, batch, vocab_size)
            else:
                output = decoder(decoder_input)
                # output: (caption_length + 1, batch, vocab_size)
            output = output[1:-1, :, :].view(-1, config.vocab_size)

            targets = caption_batch.permute(1, 0)[1:, :].reshape(-1)
            mask = targets != vocab.word2index[vocab.pad]
            # only compare non-pad tokens
            output = output[mask, :]
            targets = targets[mask]

            probs = torch.exp(output)
            acc_batch = (probs.max(dim=-1)[1] == targets).float().mean()
            acc.append(acc_batch.item())

    print('Accuracy: ', sum(acc) / len(acc))

    if args.model == 'lstm':
        torch.save(encoder.state_dict(), config.encoder_lstm_file)
        torch.save(emb_layer.state_dict(), config.embedding_lstm_file)
        torch.save(decoder.state_dict(), config.decoder_lstm_file)
    else:
        torch.save(encoder.state_dict(), config.encoder_gpt1_file)
        torch.save(emb_layer.state_dict(), config.embedding_gpt1_file)
        torch.save(decoder.state_dict(), config.decoder_gpt1_file)
