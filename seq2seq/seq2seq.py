# coding: utf-8
import os
import pickle
import numpy as np
from seq2seq.seq2seq_layers import Encoder, Decoder
from common.layers import TimeSoftmaxWithLoss


class Seq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        params = [p.astype(np.float16) for p in self.params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        if "/" in file_name:
            file_name = file_name.replace("/", os.sep)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [p.astype("f") for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
