# coding: utf-8
# import sys
# sys.path.append("/content/drive/MyDrive/nlp_demo")
import numpy as np


class RNN:
  """实现单个时间步的 RNN 单元"""
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]  # 输入权重 Wx, 循环权重 Wh, 偏置 b
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.cache = None  # 缓存前向传播中间结果
  
  def forward(self, x, h_prev):
    Wx, Wh, b = self.params
    t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
    h_next = np.tanh(t)  # 计算得到下一个隐藏状态

    self.cache = (x, h_prev, h_next)
    return h_next
  
  def backward(self, dh_next):
    Wx, Wh, b = self.params
    x, h_prev, h_next = self.cache

    dt = dh_next * (1 - h_next ** 2)  # tanh 导数
    db = np.sum(dt, axis=0)  # 偏置梯度
    dWh = np.dot(h_prev.T, dt)  # Wh 梯度
    dh_prev = np.dot(dt, Wh.T)  # 上一隐藏层梯度
    dWx = np.dot(x.T, dt)  # Wx 梯度
    dx = np.dot(dt, Wx.T)  # 输入梯度

    self.grads[0][...] = dWx
    self.grads[1][...] = dWh
    self.grads[2][...] = db

    return dx, dh_prev


class TimeRNN:
  """封装实现整个序列的 RNN 计算模块"""
  def __init__(self, Wx, Wh, b, stateful=False):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layer = None

    self.h, self.dh = None, None
    self.stateful = stateful  # 是否保持序列间隐藏状态

  def set_state(self, h):
    self.h = h
  
  def reset_state(self):
    self.h = None
  
  def forward(self, xs):
    Wx, Wh, b = self.params
    N, T, D = xs.shape
    D, H = Wx.shape

    self.layers = []
    hs = np.empty((N, T, H), dtype="f")

    if not self.stateful or self.h is None:
      self.h = np.zeros((N, H), dtype="F")
    
    # 按时间步展开
    for t in range(T):
      layer = RNN(*self.params)
      self.h = layer.forward(xs[:, t, :], self.h)
      hs[:, t, :] = self.h
      self.layers.append(layer)
    
    return hs
  
  def backward(self, dhs):
    Wx, Wh, b = self.params
    N, T, D = dhs.shape
    D, H = Wx.shape
    
    dxs = np.empty((N, T, H), dtype="f")
    dh = 0
    grads = [0, 0, 0]
    # 按时间步反向传播
    for t in reversed(range(T)):
      layer = self.layers[t]
      dx, dh = layer.backward(dhs[:, t, :] + dh)
      dxs[: ,t, :] = dx

      # 累积参数梯度
      for i, grad in enumerate(layer.grads):
        grads[i] += grad
    
    for i, grad in enumerate(grads):
      self.grads[i][...] = grad
    self.dh = dh

    return dxs
