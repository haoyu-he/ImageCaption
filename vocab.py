from nltk.tokenize import RegexpTokenizer
from collections import Counter

from config import Config


class Vocab:

    def __init__(self):

        self.counter = Counter()
        self.word2index = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        self.index2word = {
            0: '<pad>',
            1: '<sos>',
            2: '<eos>',
            3: '<unk>'
        }
        self.pad, self.sos, self.eos, self.unk = '<pad>', '<sos>', '<eos>', '<unk>'
        self.size = 4

        self.word_tokenize = RegexpTokenizer(r'(?:[a-zA-Z]+|<[a-zA-Z]+>)').tokenize

    def add_sentence(self, sentence: str):

        tokens = self.word_tokenize(sentence)
        self.counter.update(tokens)

    def init_vocab(self):

        # initialize everything
        self.counter.clear()
        self.word2index = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        self.index2word = {
            0: '<pad>',
            1: '<sos>',
            2: '<eos>',
            3: '<unk>'
        }
        self.size = 4

    def build_vocab(self, file_path: str, vocab_size: int = None):

        self.init_vocab()

        # add vocabs
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):

                # ignore header line
                if i == 0:
                    continue

                caption = line.strip().lower().split(",", 1)[1]  # id=0, caption=1
                self.add_sentence(caption)

        if vocab_size is not None:
            vocab_size -= 4
        vocabs = self.counter.most_common(vocab_size)

        for word, _ in vocabs:
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.size += 1

    def get_index(self, word: str) -> int:
        if word not in self.word2index:
            return self.word2index['<unk>']
        return self.word2index[word]

    def save_vocab(self, vocab_file: str):

        with open(vocab_file, 'a') as file:
            for word in self.word2index.keys():
                line = f'{word},{self.word2index[word]}\n'
                file.write(line)

    def load_vocab(self, vocab_file: str):

        self.init_vocab()

        with open(vocab_file, 'r') as file:
            for line in file:
                word, index = line.split(',')
                self.word2index[word] = int(index)
                self.index2word[int(index)] = word


if __name__ == '__main__':
    config = Config()

    vocab = Vocab()
    vocab.build_vocab(config.caption_file, config.vocab_size)
    vocab.save_vocab(config.vocab_file)