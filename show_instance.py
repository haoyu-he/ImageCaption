import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from config import Config
from vocab import Vocab
from model import ImageEncoder, TextDecoder


parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, default='6261030.jpg')
args = parser.parse_args()

config = Config

# load vocabulary
vocab = Vocab()
vocab.load_vocab(config.vocab_file)

# load image
image_file = os.path.join(config.image_dir, args.image_file)
image = Image.open(image_file).convert('RGB')

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_ori = transform(image).to(config.device)
image_norm = normalize(image_ori)

# load model
encoder = ImageEncoder(word_emb_dim=config.word_emb_dim).to(config.device)
emb_layer = torch.nn.Embedding(num_embeddings=config.vocab_size,
                               embedding_dim=config.word_emb_dim,
                               padding_idx=vocab.word2index[vocab.pad]).to(config.device)
decoder = TextDecoder(word_emb_dim=config.word_emb_dim,
                      hidden_dim=config.decoder_hidden_dim,
                      num_layers=config.num_decoder_layers,
                      vocab_size=config.vocab_size).to(config.device)

encoder.load_state_dict(torch.load(config.encoder_file, map_location=config.device))
emb_layer.load_state_dict(torch.load(config.embedding_file, map_location=config.device))
decoder.load_state_dict(torch.load(config.decoder_file, map_location=config.device))

encoder.eval()
emb_layer.eval()
decoder.eval()

'''generate sentence'''
image_norm = image_norm.unsqueeze(0)
# image_norm: (1, 3, 224, 224)

hidden = decoder.hidden_0
cell = decoder.cell_0

sentence = []
word_indices = torch.tensor([vocab.word2index[vocab.sos]], dtype=torch.long, device=config.device).unsqueeze(0)

# get image embedding
image_emb = encoder(image_norm).unsqueeze(0)
# image_emb: (1, 1, word_emb_dim)

for i in range(config.max_length):

    word_seq = emb_layer(word_indices).permute(1, 0, 2)
    # word_seq: (sequence_length, batch, word_emb_dim)

    decoder_input = torch.cat([image_emb, word_seq], dim=0)

    next_pred, (hidden, cell) = decoder(decoder_input, hidden, cell)
    # next_pred: (caption_length, batch, vocab_size)
    next_pred = torch.argmax(next_pred[-1, 0, :])

    word_indices = torch.cat([word_indices, next_pred.view(1, 1)], dim=-1)

    next_word = vocab.index2word[next_pred.item()]
    if next_word == vocab.eos:
        break

    sentence.append(next_word)

sentence = ' '.join(sentence).capitalize().strip()
image_id = args.image_file.split('.')[0]
plt.imshow(image_ori.cpu())
plt.title(sentence)
plt.savefig(image_id + '.pdf', bbox_inches='tight')

print(sentence)