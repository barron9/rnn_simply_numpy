#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import savetxt

class VanillaRNN:

    def __init__(self, char_to_idx, idx_to_char, vocab_size, hidden_layer_size=75,
                 seq_len=20, clip_rate=5, epochs=50, learning_rate=1e-2):
        """
        Implementation of simple character-level RNN using Numpy
        Major inspiration from karpathy/min-char-rnn.py
        """

        # assign instance variables
        self.char_to_idx = char_to_idx  # dictionary that maps characters in the vocabulary to an index
        self.idx_to_char = idx_to_char  # dictionary that maps indices to unique characters in the vocabulary

        self.vocab_size = vocab_size  # number of unique characters in the training data
        self.n_h = hidden_layer_size  # desirable number of units in the hidden layer

        self.seq_len = seq_len  # number of characters that will be fed to the RNN in each batch (also number of time steps)
        self.clip_rate = clip_rate  # maximum absolute value for the gradients, which are limited to avoid exploding gradients
        self.epochs = epochs  # number of training iterations
        self.learning_rate = learning_rate

        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len  # smoothing out loss as batch SGD is noisy

        # initialize parameters
        self.params = {}
        self.next=""
        self.params["W_xh"] = np.random.randn(self.vocab_size, self.n_h) * 0.01

        self.params["W_hh"] = np.random.randn(self.n_h, self.n_h) * 0.01
        self.params["b_h"] = np.zeros((1, self.n_h))

        self.params["W_hy"] = np.random.randn(self.n_h, self.vocab_size) * 0.01
        self.params["b_y"] = np.zeros((1, self.vocab_size))

        self.h0 = np.zeros((1, self.n_h))  # value of hidden state at time step t = -1. This is updated over time
        self.nexte=" "
        # initialize gradients and memory parameters for Adagrad
        self.grads = {}
        self.m_params = {}
        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.m_params["m" + key] = np.zeros_like(self.params[key])


    def encode_data(self, X):
        """
        Encodes input text to integers
        Input:
            X - input text (string)
        Outputs:
            X_encoded - list with encoded characters from input text
        """
        X_encoded = []

        for char in X:
            X_encoded.append(self.char_to_idx[char])
        return X_encoded


    def prepare_batches(self, X, index):
        """
        Prepares one-hot-encoded X and Y batches that will be fed into the RNN
        Input:
            X - encoded input data (string)
            index - index of batch training loop, used to create training batches
        Output:
            X_batch - one-hot-encoded list
            y_batch - similar to X_batch, but every character is shifted one time step to the right
        """
        X_batch_encoded = X[index: index + self.seq_len]
        y_batch_encoded = X[index + 1: index + self.seq_len + 1]

        X_batch = []
        y_batch = []

        for i in X_batch_encoded:
            one_hot_char = np.zeros((1, self.vocab_size))
            one_hot_char[0][i] = 1
            X_batch.append(one_hot_char)

        for j in y_batch_encoded:
            one_hot_char = np.zeros((1, self.vocab_size))
            one_hot_char[0][j] = 1
            y_batch.append(one_hot_char)# bir sonraki harfe bağlıyör
        #print(self.idx_to_char[np.argmax(X_batch[0][0])], self.idx_to_char[np.argmax(y_batch[0][0])]) 
       # exit()
        return X_batch, y_batch


    def softmax(self, x):
        """
        Normalizes output into a probability distribution
        """
        e_x = np.exp(x - np.max(x)) # max value is substracted for numerical stability
        return e_x / np.sum(e_x)


    def forward_pass(self, X):
        """
        Implements the forward propagation for a RNN
        Input:
            X - input batch, one-hot-encoded
        Output:
            y_pred - output softmax probabilities
            h - hidden states
        """
        h = {}  # stores hidden states
        h[-1] = self.h0  # set initial hidden state at t=-1
        
        y_pred = {}  # stores softmax output probabilities
        #print("X "+str(X))
        # iterate over each character in the input sequence
        for t in range(self.seq_len):
            #print(h[t - 1])
  
            h[t] = np.tanh(np.dot(X[t], self.params["W_xh"]) + np.dot(h[t - 1], self.params["W_hh"]) + self.params["b_h"])
            #print(h[t])
            #exit()
            y_pred[t] = self.softmax(np.dot(h[t], self.params["W_hy"]) + self.params["b_y"])
            #exit()
        # with open("data.wei", "r+") as f:
        #     savetxt(f, self.params["W_xh"], delimiter=',')
        #     savetxt(f, self.params["W_hh"], delimiter=',')
        #     savetxt(f, self.params["W_hy"], delimiter=',')
        #     savetxt(f, self.params["b_h"], delimiter=',')
        #     savetxt(f, self.params["b_y"], delimiter=',')

        #print(" W_xh ",self.params["W_xh"]," W_hh ",self.params["W_hh"]," b_h ",self.params["b_h"]," W_hy ",self.params["W_hy"]," b_y ",self.params["b_y"])
        #print("y_pred "+str(y_pred))
        self.ho = h[t]
        #print(h)
        #exit()
        return y_pred, h


    def backward_pass(self, X, y, y_pred, h):
        """
        Implements the backward propagation for a RNN
        Input:
            X - input batch, one-hot-encoded
            y - label batch, one-hot-encoded
            y_pred - output softmax probabilities from forward propagation
            h - hidden states from forward propagation
        Output:
            grads - derivatives of every learnable parameter in the RNN
        """
        dh_next = np.zeros_like(h[0])

        for t in reversed(range(self.seq_len)):
            dy = np.copy(y_pred[t])
            dy[0][np.argmax(y[t])] -= 1  # predicted y - actual y

            self.grads["dW_hy"] += np.dot(h[t].T, dy)
            self.grads["db_y"] += dy

            dhidden = (1 - h[t] ** 2) * (np.dot(dy, self.params["W_hy"].T) + dh_next)
            dh_next = np.dot(dhidden, self.params["W_hh"].T)

            self.grads["dW_hh"] += np.dot(h[t - 1].T, dhidden)
            self.grads["dW_xh"] += np.dot(X[t].T, dhidden)
            self.grads["db_h"] += dhidden
            #print("dhidden ",dhidden[0][0],self.grads["dW_xh"][0][0],self.grads["dW_hh"][0][0])
            #exit()
            # clip gradients to mitigate exploding gradients
        for grad, key in enumerate(self.grads):
            np.clip(self.grads[key], -self.clip_rate, self.clip_rate, out=self.grads[key])
        return


    def update_params(self):
        """
        Updates parameters with Adagrad
        """
        for key in self.params:
            self.m_params["m" + key] += self.grads["d" + key] * self.grads["d" + key]
            self.params[key] -= self.grads["d" + key] * self.learning_rate / (np.sqrt(self.m_params["m" + key]) + 1e-8)


    def sample(self, sample_size, start_index):
        """
        Outputs a sample sequence from the model
        Input:
            sample_size - desirable size of output string (in chars)
            start_index - index for first time step
        Output:
            s - string made of sampled characters, of size sample_size
        """
        s = ""
        hs = {}  # stores hidden states
        hs = self.h0  # set initial hidden state at t=-1
        x = np.zeros((1, self.vocab_size))
        #print(x.shape,x)
        #exit()
        

        for i in range(0,20,1):
            #print("importing ", self.nexte)
          
            x = np.zeros((1, self.vocab_size))
            #print(self.nexte)
            index = self.char_to_idx[self.nexte]
            #print(self.nexte,index)
            #exit() 
            #x = np.zeros((1, self.vocab_size))
            #elf.nexte=x
            x[0][index] = 1

            hs = np.tanh(np.dot(x, self.params["W_xh"]) + np.dot(hs, self.params["W_hh"]) + self.params["b_h"])
 
            y_pred = self.softmax(np.dot(hs, self.params["W_hy"]) + self.params["b_y"])

            for m in range(len(y_pred)):
                #print("lenpred",len(y_pred),self.idx_to_char[np.argmax(y_pred[i])])
                s+=self.idx_to_char[np.argmax(y_pred[m])]
                self.nexte=self.idx_to_char[np.argmax(y_pred[m])]
        #exit()
            ##print(i)
            if i == 19:
                self.nexte=" "
        print(s)
        return s


    def train(self, X, verbose=False):
        """
        Main method of the RNN class where training takes place
        Input:
            X - input text (string)
            verbose - if true, the training loss and smaple text will be shown
                      at frequent intervals
        Output:
            J - training loss at every training step
            params - final values of training parameters
        """
        J = []
 
        num_batches = len(X) // self.seq_len

        X_trimmed = X[:num_batches * self.seq_len] # trim end of the input text so that we have full sequences
        
        X_encoded = self.encode_data(X_trimmed) # transform words to indices to enable processing

        #exit()
        for i in range(self.epochs):

            for j in range(0, len(X_encoded) - self.seq_len, self.seq_len):

                X_batch, y_batch = self.prepare_batches(X_encoded, j)
                #print(X_batch[0][0])
                #exit()
                y_pred, h = self.forward_pass(X_batch)
                
                loss = 0
                s=""
                r=""
                for t in range(self.seq_len):
                    s+=(self.idx_to_char[np.argmax(y_pred[t])])
                    r+=(self.idx_to_char[np.argmax(y_batch[t])])
                    #print(t," seq ",np.log(y_pred[t][0, np.argmax(y_batch[t])]))
                    loss += -np.log(y_pred[t][0, np.argmax(y_batch[t])])
                #print(s)
                #exit()
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                J.append(self.smooth_loss)

                self.backward_pass(X_batch, y_batch, y_pred, h)
                self.update_params()

            
                #print(s,"|",r)
            if  i+1 > 299:
               self.sample(25,0)
                #exit()
            (self.sample(10,0))
            if verbose:
                print('Epoch:', i + 1, "\tLoss:", loss, "")
           
        return J, self.params
