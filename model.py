"""Implement a LSTM network from scratch using numpy."""

__author__ = ' Simone Carolini'

import numpy as np
from typing import Type, List, Union, Any, Tuple, Dict


class LSTM(object):
    """Implementation of a LSTM that learns how to sum by 4."""

    def __init__(self, char_to_ind: Type[Dict], ind_to_char: Type[Dict], vocab_size: int = None, n_h=100, seq_len=40,
                 epochs=10, lr=0.001, beta1=0.9, beta2=0.999):

        self.char_to_ind = char_to_ind  # char to ind mapping.
        self.ind_to_char = ind_to_char  # ind to character mapping.
        self.vocab_size = vocab_size  # number of characters in the data set.
        self.n_h = n_h  # neurons in the hidden layer.
        self.seq_len = seq_len
        self.epochs = epochs  # no. of training iterations.
        self.lr = lr  # learning rate.
        self.beta1 = beta1  # 1st momentum Adam parameter.
        self.beta2 = beta2  # 2nd momentum Adam parameter.

        # Params to initialize (Weights and bias)
        self.params = {}
        xavier_std = (1.0 / np.sqrt(self.vocab_size + self.n_h))  # Xavier initialisation
        # Forget gate layer
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * xavier_std
        self.params["bf"] = np.ones((self.n_h, 1))
        # Input gate layer
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * xavier_std
        self.params["bi"] = np.zeros((self.n_h, 1))
        # Cell gate layer
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * xavier_std
        self.params["bc"] = np.zeros((self.n_h, 1))
        # Output Gate layer
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * xavier_std
        self.params["bo"] = np.zeros((self.n_h, 1))
        # Output
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * (1.0 / np.sqrt(self.vocab_size))
        self.params["bv"] = np.zeros((self.vocab_size, 1))
        # Adam and gradients
        self.grads = {}
        self.adam_params = {}
        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.adam_params["m" + key] = np.zeros_like(self.params[key])
            self.adam_params["v" + key] = np.zeros_like(self.params[key])
        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len
        return

    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax formula."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def _clip_grads(self):
        """
        Limits the magnitude of gradients to avoid exploding gradients
        """
        for key in self.grads:
            np.clip(self.grads[key], -5, 5, out=self.grads[key])
        return

    def _reset_grads(self):
        """Resets gradients to zero at each new back-propagation"""
        for key in self.grads:
            self.grads[key].fill(0)
        return

    def adam(self, batch_num):
        """
        Updates parameters with Adam
        """
        for key in self.params:
            self.adam_params["m" + key] = self.adam_params["m" + key] * self.beta1 + \
                                          (1 - self.beta1) * self.grads["d" + key]
            self.adam_params["v" + key] = self.adam_params["v" + key] * self.beta2 + \
                                          (1 - self.beta2) * self.grads["d" + key] ** 2

            m_correlated = self.adam_params["m" + key] / (1 - self.beta1 ** batch_num)
            v_correlated = self.adam_params["v" + key] / (1 - self.beta2 ** batch_num)
            self.params[key] -= self.lr * m_correlated / (np.sqrt(v_correlated) + 1e-8)
        return

    def _sample(self, h_prev, c_prev, sample_size):
        """
        Outputs a sample sequence from the model
        """
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        sample_string = ""

        for t in range(sample_size):
            y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)

            # get a random index within the probability distribution of y_hat(ravel())
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # find the char with the sampled index and concat to the output string
            char = self.ind_to_char[idx]
            sample_string += char
        return sample_string

    def _forward_step(self, x, h_prev, c_prev):
        """
        Implements the forward propagation for one time step
        """
        z = np.row_stack((h_prev, x))

        f = self.sigmoid(np.dot(self.params["Wf"], z) + self.params["bf"])
        i = self.sigmoid(np.dot(self.params["Wi"], z) + self.params["bi"])
        c_bar = np.tanh(np.dot(self.params["Wc"], z) + self.params["bc"])

        c = f * c_prev + i * c_bar
        o = self.sigmoid(np.dot(self.params["Wo"], z) + self.params["bo"])
        h = o * np.tanh(c)

        v = np.dot(self.params["Wv"], h) + self.params["bv"]
        y_hat = self.softmax(v)
        return y_hat, v, h, o, c, c_bar, i, f, z

    def _backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
        """
        Implements the backward propagation for one time step
        """
        dv = np.copy(y_hat)
        dv[y] -= 1  # yhat - y

        self.grads["dWv"] += np.dot(dv, h.T)
        self.grads["dbv"] += dv

        dh = np.dot(self.params["Wv"].T, dv)
        dh += dh_next

        do = dh * np.tanh(c)
        da_o = do * o * (1 - o)
        self.grads["dWo"] += np.dot(da_o, z.T)
        self.grads["dbo"] += da_o

        dc = dh * o * (1 - np.tanh(c) ** 2)
        dc += dc_next

        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar ** 2)
        self.grads["dWc"] += np.dot(da_c, z.T)
        self.grads["dbc"] += da_c

        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads["dWi"] += np.dot(da_i, z.T)
        self.grads["dbi"] += da_i

        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads["dWf"] += np.dot(da_f, z.T)
        self.grads["dbf"] += da_f

        dz = (np.dot(self.params["Wf"].T, da_f)
              + np.dot(self.params["Wi"].T, da_i)
              + np.dot(self.params["Wc"].T, da_c)
              + np.dot(self.params["Wo"].T, da_o))

        dh_prev = dz[:self.n_h, :]
        dc_prev = f * dc
        return dh_prev, dc_prev

    def _forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        """
        Implements the forward and backward propagation for one batch
        """
        x, z = {}, {}
        f, i, c_bar, c, o = {}, {}, {}, {}, {}
        y_hat, v, h = {}, {}, {}

        # Values at t= - 1
        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0
        for t in range(self.seq_len):
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][x_batch[t]] = 1

            y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = \
                self.forward_step(x[t], h[t - 1], c[t - 1])

            loss += -np.log(y_hat[t][y_batch[t], 0])

        self.reset_grads()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_len)):
            dh_next, dc_next = self._backward_step(y_batch[t], y_hat[t], dh_next, dc_next, c[t - 1], z[t], f[t], i[t],
                                                   c_bar[t], c[t], o[t], h[t])
        return loss, h[self.seq_len - 1], c[self.seq_len - 1]

    def train(self, X, verbose=True):
        """
        Main method of the LSTM class where training takes place
        """
        J = []  # to store losses

        num_batches = len(X) // self.seq_len
        X = X[: num_batches * self.seq_len]  # trim input to have full sequences

        for epoch in range(self.epochs):
            h_prev = np.zeros((self.n_h, 1))
            c_prev = np.zeros((self.n_h, 1))

            for j in range(0, len(X) - self.seq_len, self.seq_len):
                # prepare batches
                x_batch = [self.char_to_ind[ch] for ch in X[j: j + self.seq_len]]
                y_batch = [self.char_to_ind[ch] for ch in X[j + 1: j + self.seq_len + 1]]

                loss, h_prev, c_prev = self._forward_backward(x_batch, y_batch, h_prev, c_prev)

                # smooth out loss and store in list
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                J.append(self.smooth_loss)

                self._clip_grads()

                batch_num = epoch * self.epochs + j / self.seq_len + 1
                self.adam(batch_num)

                # print out loss and sample string
                if verbose:
                    if j % 400000 == 0:
                        print('Epoch:', epoch, '\tBatch:', j, "-", j + self.seq_len,
                              '\tLoss:', round(self.smooth_loss, 2))
                        s = self._sample(h_prev, c_prev, sample_size=250)
                        print(s, "\n")

        return J, self.params
