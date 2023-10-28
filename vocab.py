import os
import string
import nltk

from collections import Counter


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
        self.size = 4

    def add_sentence(self, sentence: str):

        tokens = nltk.tokenize.word_tokenize(sentence)
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
