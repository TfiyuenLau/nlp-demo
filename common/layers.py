# coding: utf-8
import numpy as np
from common.functions import softmax


class Embedding:
    """将离散的单词 ID 映射到稠密的向量表示"""

    def __init__(self, W):
        self.params = [W]  # 词向量矩阵(V, D)
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        (W,) = self.params
        self.idx = idx
        out = W[idx]  # 取出对应词向量
        return out

    def backward(self, dout):
        (dW,) = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)  # 在对应的 idx 位置累积梯度
        return None


class TimeEmbedding:
    """在每个时间步调用一个 Embedding 层"""

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        """
        xs: 输入形状 (N, T)，N=批大小，T=时间步数
        输出: (N, T, D)，每个词 ID 被映射为 D 维词向量
        """
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        # 遍历每个时间步，调用单步 Embedding
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        """
        dout: 上游梯度 (N, T, D)
        返回: None（因为输入是离散 ID，没有梯度）
        """
        N, T, D = dout.shape

        grad = 0
        # 按时间步反向传播
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    """全连接层"""

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class TimeSoftmaxWithLoss:
    """计算整个序列 (N, T) 的分类损失，支持忽略标签"""

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        """
        xs: (N, T, V)，预测值(未归一化 logits)
        ts: (N, T) 或 (N, T, V)，标签(int 或 one-hot)

        return：平均交叉熵损失
        """
        N, T, V = xs.shape

        # 如果标签是 one-hot，则转换为类别索引
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)

        mask = ts != self.ignore_label

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        # softmax操作并交叉熵损失
        ys = softmax(xs)  # (N*T, V)
        ls = np.log(ys[np.arange(N * T), ts] + 1e-7)  # 取出正确类别的 log 概率
        ls *= mask  # 忽略标记的部分置 0
        loss = -np.sum(ls) / mask.sum()  # 平均损失

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        """
        反向传播：计算 softmax 的输入梯度 dx
        dout: 上游梯度（默认 1）
        返回：dx，形状 (N, T, V)
        """
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()  # 平均化
        dx *= mask[:, np.newaxis]  # 忽略无效标签对应的梯度

        dx = dx.reshape((N, T, V))
        return dx


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class LayerNorm:
    """层归一化（Layer Normalization）

    在最后一个维度（特征维度）上进行归一化，使每个样本的特征均值为0，方差为1。
    包含可学习的缩放参数gamma和偏移参数beta。

    参数:
      d_model: 特征维度（归一化的维度）
      eps: 防止除零的小常数
    """

    def __init__(self, d_model, eps=1e-6):
        # 可学习参数：gamma（缩放）和 beta（偏移）
        self.gamma = np.ones(d_model, dtype="f")
        self.beta = np.zeros(d_model, dtype="f")
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

        self.eps = eps
        self.cache = None

    def forward(self, x):
        """
        前向传播
        x: 输入，形状可以是 (N, D) 或 (N, T, D)，在最后一维进行归一化
        返回: 归一化后的输出，形状与输入相同
        """
        # 计算均值和方差（在最后一个维度上）
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移
        out = self.gamma * x_normalized + self.beta

        # 缓存用于反向传播
        self.cache = (x, x_normalized, mean, var)
        return out

    def backward(self, dout):
        """
        反向传播
        dout: 上游梯度，形状与forward的输出相同
        返回: 对输入x的梯度
        """
        x, x_normalized, mean, var = self.cache
        gamma = self.params[0]

        # 获取维度信息
        original_shape = x.shape
        D = original_shape[-1]  # 特征维度
        N = np.prod(original_shape[:-1])  # 批次大小（可能包含时间步）  # noqa: F841

        # gamma 和 beta 的梯度
        dgamma = np.sum(dout * x_normalized, axis=tuple(range(len(original_shape) - 1)))
        dbeta = np.sum(dout, axis=tuple(range(len(original_shape) - 1)))

        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta

        # 对输入 x 的梯度
        dx_normalized = dout * gamma

        # 标准差的倒数
        std_inv = 1.0 / np.sqrt(var + self.eps)

        # 计算 x 的梯度（考虑均值和方差的依赖关系）
        dx = (
            (1.0 / D)
            * std_inv
            * (
                D * dx_normalized
                - np.sum(dx_normalized, axis=-1, keepdims=True)
                - x_normalized
                * np.sum(dx_normalized * x_normalized, axis=-1, keepdims=True)
            )
        )

        return dx


class PositionalEncoding:
    """位置编码（Positional Encoding）

    使用正弦和余弦函数为序列中的每个位置生成固定的位置编码。
    位置编码与词嵌入相加，为模型提供序列中的位置信息。

    公式:
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数:
      d_model: 模型的特征维度
      max_len: 支持的最大序列长度
    """

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model

        # 生成位置编码矩阵 (max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype="f")

        # 位置索引 (max_len, 1)
        position = np.arange(0, max_len, dtype="f").reshape(-1, 1)

        # 分母项: 10000^(2i/d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype="f") * -(np.log(10000.0) / d_model)
        )

        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe  # 位置编码表 (max_len, d_model)

        # 位置编码是固定的，不需要训练
        self.params, self.grads = [], []

    def forward(self, x):
        """
        前向传播：将位置编码添加到输入中

        x: 输入张量，形状 (N, T, D)
           N = 批次大小
           T = 序列长度
           D = 特征维度（必须等于d_model）

        返回: 添加位置编码后的张量，形状 (N, T, D)
        """
        N, T, D = x.shape
        assert D == self.d_model, f"输入特征维度 {D} 与模型维度 {self.d_model} 不匹配"
        assert T <= self.pe.shape[0], f"序列长度 {T} 超过最大长度 {self.pe.shape[0]}"

        # 取出对应长度的位置编码 (T, D)，通过广播加到 (N, T, D)
        pos_encoding = self.pe[:T, :]
        out = x + pos_encoding

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T, D)
        返回: 对输入x的梯度，形状 (N, T, D)

        位置编码是固定的常量，不需要计算梯度，直接传递梯度即可
        """
        return dout
