# coding: utf-8
import os
import numpy as np
import pickle
from common.layers import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss, TimeDropout
from lstm.lstm_layers import TimeLSTM


class GatedRnnlm:
    """基于LSTM实现语言模型"""

    def __init__(
        self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Xavier 初始化方法
        rn = np.random.randn
        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1 = np.zeros(4 * H).astype("f")
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2 = np.zeros(4 * H).astype("f")
        affine_b = np.zeros(V).astype("f")

        # 网络模型结构
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),  # 优化：Dropout
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b),  # 优化：权重共享
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

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
