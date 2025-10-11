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
    W, = self.params
    self.idx = idx
    out = W[idx]  # 取出对应词向量
    return out

  def backward(self, dout):
    dW, = self.grads
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

    out = np.empty((N, T, D), dtype='f')
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
    
    mask = (ts != self.ignore_label)

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
