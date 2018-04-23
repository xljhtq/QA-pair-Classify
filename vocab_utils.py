# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import re
import os


# import math
class Vocab(object):
    def fromText_format3(self, train_dir, wordvec_path):
        vec_path = os.path.join(train_dir, wordvec_path)
        self.word2id = {}
        self.id2word = {}

        vec_file = open(vec_path, 'rt')
        word_vecs = {}
        for line in vec_file:
            line = line.strip()
            parts = line.split(' ')
            word = parts[0]
            self.word_dim = len(parts[1:])
            if self.word_dim < 128: continue
            vector = np.array(parts[1:], dtype='float32')
            cur_index = len(self.word2id)
            self.word2id[word] = cur_index
            self.id2word[cur_index] = word
            word_vecs[cur_index] = vector
        vec_file.close()
        cur_index = len(self.word2id)
        self.word2id['<UNK/>'] = cur_index
        self.id2word[cur_index] = '<UNK/>'
        word_vecs[cur_index] = np.random.normal(0, 1, size=(self.word_dim,))
        self.vocab_size = len(self.word2id)

        self.word_vecs = np.zeros((self.vocab_size, self.word_dim),
                                  dtype=np.float32)
        for cur_index in iter(range(self.vocab_size)):
            self.word_vecs[cur_index][:len(word_vecs[cur_index])] = word_vecs[cur_index]

        word2id_path = os.path.join(train_dir, "data/word2id")
        with open(word2id_path, "w") as out_op:
            for word in self.word2id:
                out_op.write(word + "\t" + str(self.word2id[word]) + "\n")

    def to_index_sequence(self, sentence):
        sentence = sentence.strip()
        seq = []
        for word in re.split('\\s+', sentence):
            if word in self.word2id:
                idx = self.word2id[word]
            else:
                idx = self.word2id['<UNK/>']
            seq.append(idx)
        return seq
