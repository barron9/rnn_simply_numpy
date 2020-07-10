#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

from model import VanillaRNN

def main():

    # ----DATA PREPARATION---- #
    data = open('/Users/berkintatlisu/Downloads/scratchML-master/RNN/HP1.txt').read().lower()
    #data = data[:500] # reduce data to enable faster processing
    #print(data)
    chars = set(data)
    #exit()

    vocab_size = len(chars)

    # creating dictionaries for mapping chars to ints and vice versa
    char_to_idx = {w: i for i, w in enumerate(chars)}
    idx_to_char = {i: w for i, w in enumerate(chars)}

    # ----MODEL TRAINING---- #
    hyperparameters = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'hidden_layer_size': 75,
        'seq_len': 25,  # keep small to avoid diminishing / exploding gradients
        'clip_rate': 5,
        'epochs': 1300,
        'learning_rate': 1e-2,
    }

    model = VanillaRNN(**hyperparameters)

    J, params = model.train(data, verbose = True)

    plt.plot([i for i in range(len(J))], J)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()

    # get sample text

if __name__ == "__main__":
    main()
