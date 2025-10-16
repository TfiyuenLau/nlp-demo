# coding: utf-8
import numpy as np
from common.layers import Softmax


class ScaledDotProductAttention:
    """缩放点积注意力（Scaled Dot-Product Attention）

    Transformer的核心注意力机制，计算公式：
      Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    参数:
      d_k: Key的维度，用于缩放
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
        self.attention_weights = None

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        Q: Query矩阵，形状 (N, T_q, d_k)
        K: Key矩阵，形状 (N, T_k, d_k)
        V: Value矩阵，形状 (N, T_k, d_v)
        mask: 可选的掩码，形状 (N, T_q, T_k) 或 (N, 1, T_k)
              mask中为True的位置会被设置为-inf（在softmax前）

        返回: 注意力输出，形状 (N, T_q, d_v)
        """
        N, T_q, d_k = Q.shape
        N, T_k, d_v = V.shape

        # 计算注意力分数: Q·K^T，形状 (N, T_q, T_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1))

        # 缩放，防止softmax饱和
        scale = np.sqrt(d_k)
        scores = scores / scale

        # 应用掩码（如果提供）
        if mask is not None:
            # 将mask为True的位置设为一个很大的负数
            scores = np.where(mask, -1e9, scores)

        # Softmax归一化，得到注意力权重
        # 需要对每个query位置的所有key做softmax
        N, T_q, T_k = scores.shape
        scores_2d = scores.reshape(N * T_q, T_k)
        attention_weights = self.softmax.forward(scores_2d)
        attention_weights = attention_weights.reshape(N, T_q, T_k)

        # 加权求和: attention_weights · V，形状 (N, T_q, d_v)
        out = np.matmul(attention_weights, V)

        # 缓存用于反向传播
        self.cache = (Q, K, V, attention_weights, scale, mask)
        self.attention_weights = attention_weights

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T_q, d_v)
        返回: dQ, dK, dV
        """
        Q, K, V, attention_weights, scale, mask = self.cache
        N, T_q, d_k = Q.shape
        N, T_k, d_v = V.shape

        # dout对V的梯度: dV = attention_weights^T · dout
        # attention_weights: (N, T_q, T_k), dout: (N, T_q, d_v)
        # dV: (N, T_k, d_v)
        dV = np.matmul(attention_weights.transpose(0, 2, 1), dout)

        # dout对attention_weights的梯度: d_att = dout · V^T
        # dout: (N, T_q, d_v), V: (N, T_k, d_v)
        # d_att: (N, T_q, T_k)
        d_attention_weights = np.matmul(dout, V.transpose(0, 2, 1))

        # softmax的反向传播
        d_attention_weights_2d = d_attention_weights.reshape(N * T_q, T_k)
        dscores_2d = self.softmax.backward(d_attention_weights_2d)
        dscores = dscores_2d.reshape(N, T_q, T_k)

        # 应用mask的梯度（mask位置的梯度为0）
        if mask is not None:
            dscores = np.where(mask, 0, dscores)

        # 缩放的反向传播
        dscores = dscores / scale

        # Q·K^T 的反向传播
        # scores = Q·K^T
        # dQ = dscores · K, 形状 (N, T_q, d_k)
        dQ = np.matmul(dscores, K)

        # dK = dscores^T · Q, 形状 (N, T_k, d_k)
        dK = np.matmul(dscores.transpose(0, 2, 1), Q)

        return dQ, dK, dV


class MultiHeadAttention:
    """多头注意力机制（Multi-Head Attention）

    将输入投影到多个子空间，在每个子空间独立计算注意力，然后拼接结果。

    公式:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
      其中 head_i = Attention(Q·W_qi, K·W_ki, V·W_vi)

    参数:
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        rn = np.random.randn

        # Q, K, V 的投影权重矩阵
        self.W_q = (rn(d_model, d_model) / np.sqrt(d_model)).astype("f")
        self.W_k = (rn(d_model, d_model) / np.sqrt(d_model)).astype("f")
        self.W_v = (rn(d_model, d_model) / np.sqrt(d_model)).astype("f")

        # 输出投影权重矩阵
        self.W_o = (rn(d_model, d_model) / np.sqrt(d_model)).astype("f")

        self.params = [self.W_q, self.W_k, self.W_v, self.W_o]
        self.grads = [np.zeros_like(w) for w in self.params]

        # 缩放点积注意力层
        self.attention = ScaledDotProductAttention()

        self.cache = None
        self.attention_weights = None

    def _split_heads(self, x):
        """
        将输入分割成多个头

        x: 形状 (N, T, d_model)
        返回: 形状 (N, num_heads, T, d_k)
        """
        N, T, d_model = x.shape
        # 重塑为 (N, T, num_heads, d_k)
        x = x.reshape(N, T, self.num_heads, self.d_k)
        # 转置为 (N, num_heads, T, d_k)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _combine_heads(self, x):
        """
        合并多个头的输出

        x: 形状 (N, num_heads, T, d_k)
        返回: 形状 (N, T, d_model)
        """
        N, num_heads, T, d_k = x.shape
        # 转置为 (N, T, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # 重塑为 (N, T, d_model)
        x = x.reshape(N, T, self.d_model)
        return x

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        Q: Query，形状 (N, T_q, d_model)
        K: Key，形状 (N, T_k, d_model)
        V: Value，形状 (N, T_k, d_model)
        mask: 可选的掩码，形状 (N, T_q, T_k)

        返回: 多头注意力输出，形状 (N, T_q, d_model)
        """
        N, T_q, _ = Q.shape
        N, T_k, _ = K.shape

        # 线性投影: (N, T, d_model) -> (N, T, d_model)
        Q_proj = np.matmul(Q, self.W_q)
        K_proj = np.matmul(K, self.W_k)
        V_proj = np.matmul(V, self.W_v)

        # 分割成多个头: (N, T, d_model) -> (N, num_heads, T, d_k)
        Q_split = self._split_heads(Q_proj)
        K_split = self._split_heads(K_proj)
        V_split = self._split_heads(V_proj)

        # 对每个头计算注意力
        # 将多头维度合并到批次维度: (N, num_heads, T, d_k) -> (N*num_heads, T, d_k)
        Q_split = Q_split.reshape(N * self.num_heads, T_q, self.d_k)
        K_split = K_split.reshape(N * self.num_heads, T_k, self.d_k)
        V_split = V_split.reshape(N * self.num_heads, T_k, self.d_k)

        # 调整mask的形状以适配多头
        if mask is not None:
            # mask: (N, T_q, T_k) -> (N*num_heads, T_q, T_k)
            mask = np.repeat(mask, self.num_heads, axis=0)

        # 计算注意力: (N*num_heads, T_q, d_k)
        attention_out = self.attention.forward(Q_split, K_split, V_split, mask)

        # 恢复形状: (N*num_heads, T_q, d_k) -> (N, num_heads, T_q, d_k)
        attention_out = attention_out.reshape(N, self.num_heads, T_q, self.d_k)

        # 合并多头: (N, num_heads, T_q, d_k) -> (N, T_q, d_model)
        concat = self._combine_heads(attention_out)

        # 输出投影: (N, T_q, d_model) -> (N, T_q, d_model)
        out = np.matmul(concat, self.W_o)

        # 缓存用于反向传播
        self.cache = (
            Q,
            K,
            V,
            Q_proj,
            K_proj,
            V_proj,
            Q_split,
            K_split,
            V_split,
            concat,
            mask,
        )
        self.attention_weights = self.attention.attention_weights

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T_q, d_model)
        返回: dQ, dK, dV，形状分别为 (N, T_q, d_model), (N, T_k, d_model), (N, T_k, d_model)
        """
        Q, K, V, Q_proj, K_proj, V_proj, Q_split, K_split, V_split, concat, mask = (
            self.cache
        )
        N, T_q, _ = Q.shape
        N, T_k, _ = K.shape

        # 输出投影的反向传播
        dconcat = np.matmul(dout, self.W_o.T)
        dW_o = np.matmul(
            concat.reshape(-1, self.d_model).T, dout.reshape(-1, self.d_model)
        )
        self.grads[3][...] = dW_o

        # 分割多头: (N, T_q, d_model) -> (N, num_heads, T_q, d_k)
        dconcat = dconcat.reshape(N, T_q, self.num_heads, self.d_k)
        dconcat = dconcat.transpose(0, 2, 1, 3)

        # 合并到批次维度: (N, num_heads, T_q, d_k) -> (N*num_heads, T_q, d_k)
        d_attention_out = dconcat.reshape(N * self.num_heads, T_q, self.d_k)

        # 注意力的反向传播
        dQ_split, dK_split, dV_split = self.attention.backward(d_attention_out)

        # 恢复形状: (N*num_heads, T, d_k) -> (N, num_heads, T, d_k)
        dQ_split = dQ_split.reshape(N, self.num_heads, T_q, self.d_k)
        dK_split = dK_split.reshape(N, self.num_heads, T_k, self.d_k)
        dV_split = dV_split.reshape(N, self.num_heads, T_k, self.d_k)

        # 合并多头: (N, num_heads, T, d_k) -> (N, T, d_model)
        dQ_proj = self._combine_heads(dQ_split)
        dK_proj = self._combine_heads(dK_split)
        dV_proj = self._combine_heads(dV_split)

        # 投影权重的反向传播
        dQ = np.matmul(dQ_proj, self.W_q.T)
        dK = np.matmul(dK_proj, self.W_k.T)
        dV = np.matmul(dV_proj, self.W_v.T)

        dW_q = np.matmul(
            Q.reshape(-1, self.d_model).T, dQ_proj.reshape(-1, self.d_model)
        )
        dW_k = np.matmul(
            K.reshape(-1, self.d_model).T, dK_proj.reshape(-1, self.d_model)
        )
        dW_v = np.matmul(
            V.reshape(-1, self.d_model).T, dV_proj.reshape(-1, self.d_model)
        )

        self.grads[0][...] = dW_q
        self.grads[1][...] = dW_k
        self.grads[2][...] = dW_v

        return dQ, dK, dV


class PositionwiseFeedForward:
    """位置wise前馈神经网络（Position-wise Feed-Forward Network）

    Transformer中的前馈网络，对每个位置独立应用相同的两层全连接网络。

    公式:
      FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2

    结构: d_model -> d_ff -> d_model

    参数:
      d_model: 模型的特征维度
      d_ff: 前馈网络的隐藏层维度（通常是d_model的4倍）
    """

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        rn = np.random.randn

        # 第一层权重和偏置: d_model -> d_ff
        self.W1 = (rn(d_model, d_ff) / np.sqrt(d_model)).astype("f")
        self.b1 = np.zeros(d_ff, dtype="f")

        # 第二层权重和偏置: d_ff -> d_model
        self.W2 = (rn(d_ff, d_model) / np.sqrt(d_ff)).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")

        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads = [np.zeros_like(p) for p in self.params]

        self.cache = None

    def forward(self, x):
        """
        前向传播

        x: 输入，形状 (N, T, d_model)
        返回: 输出，形状 (N, T, d_model)
        """
        N, T, d_model = x.shape

        # 第一层: x·W1 + b1，形状 (N, T, d_ff)
        # 重塑为2D进行矩阵乘法
        x_2d = x.reshape(N * T, d_model)
        hidden = np.dot(x_2d, self.W1) + self.b1

        # ReLU激活
        hidden_activated = np.maximum(0, hidden)

        # 第二层: hidden·W2 + b2，形状 (N, T, d_model)
        out_2d = np.dot(hidden_activated, self.W2) + self.b2
        out = out_2d.reshape(N, T, d_model)

        # 缓存用于反向传播
        self.cache = (x, hidden, hidden_activated)

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T, d_model)
        返回: 对输入x的梯度，形状 (N, T, d_model)
        """
        x, hidden, hidden_activated = self.cache
        N, T, d_model = x.shape

        # 重塑梯度为2D
        dout_2d = dout.reshape(N * T, self.d_model)

        # 第二层的反向传播
        dhidden_activated = np.dot(dout_2d, self.W2.T)
        dW2 = np.dot(hidden_activated.T, dout_2d)
        db2 = np.sum(dout_2d, axis=0)

        # ReLU的反向传播
        dhidden = dhidden_activated.copy()
        dhidden[hidden <= 0] = 0

        # 第一层的反向传播
        x_2d = x.reshape(N * T, d_model)
        dx_2d = np.dot(dhidden, self.W1.T)
        dW1 = np.dot(x_2d.T, dhidden)
        db1 = np.sum(dhidden, axis=0)

        # 重塑回3D
        dx = dx_2d.reshape(N, T, d_model)

        # 保存梯度
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2

        return dx


class ResidualConnection:
    """残差连接（Residual Connection）

    实现残差连接和层归一化的组合: LayerNorm(x + Sublayer(x))

    参数:
      d_model: 模型的特征维度
      dropout_ratio: dropout比率
    """

    def __init__(self, d_model, dropout_ratio=0.1):
        from common.layers import LayerNorm, TimeDropout

        self.layer_norm = LayerNorm(d_model)
        self.dropout = TimeDropout(dropout_ratio) if dropout_ratio > 0 else None

        self.params = self.layer_norm.params
        self.grads = self.layer_norm.grads

        self.cache = None

    def forward(self, x, sublayer_output):
        """
        前向传播

        x: 原始输入，形状 (N, T, d_model)
        sublayer_output: 子层的输出，形状 (N, T, d_model)
        返回: LayerNorm(x + sublayer_output)，形状 (N, T, d_model)
        """
        # 残差连接
        residual = x + sublayer_output

        # Dropout（可选）
        if self.dropout is not None:
            residual = self.dropout.forward(residual)

        # 层归一化
        out = self.layer_norm.forward(residual)

        self.cache = x
        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T, d_model)
        返回: dx（对原始输入的梯度）, dsublayer（对子层输出的梯度）
        """
        # 层归一化的反向传播
        dresidual = self.layer_norm.backward(dout)

        # Dropout的反向传播
        if self.dropout is not None:
            dresidual = self.dropout.backward(dresidual)

        # 残差连接的反向传播（梯度直接相加）
        dx = dresidual
        dsublayer = dresidual

        return dx, dsublayer


class TransformerEncoderLayer:
    """Transformer编码器层

    包含两个子层：
      1. 多头自注意力层（Multi-Head Self-Attention）
      2. 位置wise前馈网络（Position-wise Feed-Forward）

    每个子层后面都有残差连接和层归一化。

    结构:
      x -> [Multi-Head Self-Attention] -> Add & Norm ->
        -> [Feed-Forward] -> Add & Norm -> output

    参数:
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
      d_ff: 前馈网络的隐藏层维度
      dropout_ratio: dropout比率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_ratio=0.1):
        self.d_model = d_model

        # 多头自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # 残差连接和层归一化（两个）
        self.residual1 = ResidualConnection(d_model, dropout_ratio)
        self.residual2 = ResidualConnection(d_model, dropout_ratio)

        # 收集所有参数和梯度
        self.params = []
        self.grads = []

        for layer in [
            self.self_attention,
            self.feed_forward,
            self.residual1,
            self.residual2,
        ]:
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, x, mask=None):
        """
        前向传播

        x: 输入，形状 (N, T, d_model)
        mask: 可选的掩码，形状 (N, T, T)

        返回: 编码器层输出，形状 (N, T, d_model)
        """
        # 子层1: 多头自注意力
        # 自注意力中，Q=K=V=x
        attention_out = self.self_attention.forward(x, x, x, mask)

        # 残差连接和层归一化
        out1 = self.residual1.forward(x, attention_out)

        # 子层2: 前馈网络
        ff_out = self.feed_forward.forward(out1)

        # 残差连接和层归一化
        out2 = self.residual2.forward(out1, ff_out)

        # 缓存用于反向传播
        self.cache = (x, attention_out, out1, ff_out)

        return out2

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T, d_model)
        返回: 对输入x的梯度，形状 (N, T, d_model)
        """
        x, attention_out, out1, ff_out = self.cache

        # 第二个残差连接的反向传播
        dout1_1, dff_out = self.residual2.backward(dout)

        # 前馈网络的反向传播
        dout1_2 = self.feed_forward.backward(dff_out)

        # 合并梯度
        dout1 = dout1_1 + dout1_2

        # 第一个残差连接的反向传播
        dx_1, dattention_out = self.residual1.backward(dout1)

        # 自注意力的反向传播
        # 因为 Q=K=V=x，所以梯度需要累加
        dQ, dK, dV = self.self_attention.backward(dattention_out)
        dx_2 = dQ + dK + dV

        # 合并梯度
        dx = dx_1 + dx_2

        return dx


class TransformerDecoderLayer:
    """Transformer解码器层

    包含三个子层：
      1. 多头自注意力层（Masked Multi-Head Self-Attention）
      2. 多头交叉注意力层（Multi-Head Cross-Attention）
      3. 位置wise前馈网络（Position-wise Feed-Forward）

    每个子层后面都有残差连接和层归一化。

    结构:
      x -> [Masked Multi-Head Self-Attention] -> Add & Norm ->
        -> [Multi-Head Cross-Attention with encoder output] -> Add & Norm ->
        -> [Feed-Forward] -> Add & Norm -> output

    参数:
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
      d_ff: 前馈网络的隐藏层维度
      dropout_ratio: dropout比率
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_ratio=0.1):
        self.d_model = d_model

        # 多头自注意力层（带掩码）
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 多头交叉注意力层（Query来自解码器，Key和Value来自编码器）
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # 残差连接和层归一化（三个）
        self.residual1 = ResidualConnection(d_model, dropout_ratio)
        self.residual2 = ResidualConnection(d_model, dropout_ratio)
        self.residual3 = ResidualConnection(d_model, dropout_ratio)

        # 收集所有参数和梯度
        self.params = []
        self.grads = []

        for layer in [
            self.self_attention,
            self.cross_attention,
            self.feed_forward,
            self.residual1,
            self.residual2,
            self.residual3,
        ]:
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        """
        前向传播

        x: 解码器输入，形状 (N, T_dec, d_model)
        enc_output: 编码器输出，形状 (N, T_enc, d_model)
        self_mask: 自注意力掩码（因果掩码），形状 (N, T_dec, T_dec)
        cross_mask: 交叉注意力掩码（padding掩码），形状 (N, T_dec, T_enc)

        返回: 解码器层输出，形状 (N, T_dec, d_model)
        """
        # 子层1: 带掩码的多头自注意力
        # 自注意力中，Q=K=V=x
        self_attention_out = self.self_attention.forward(x, x, x, self_mask)

        # 残差连接和层归一化
        out1 = self.residual1.forward(x, self_attention_out)

        # 子层2: 多头交叉注意力
        # Q来自解码器，K和V来自编码器
        cross_attention_out = self.cross_attention.forward(
            out1, enc_output, enc_output, cross_mask
        )

        # 残差连接和层归一化
        out2 = self.residual2.forward(out1, cross_attention_out)

        # 子层3: 前馈网络
        ff_out = self.feed_forward.forward(out2)

        # 残差连接和层归一化
        out3 = self.residual3.forward(out2, ff_out)

        # 缓存用于反向传播
        self.cache = (
            x,
            enc_output,
            self_attention_out,
            out1,
            cross_attention_out,
            out2,
            ff_out,
        )

        return out3

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T_dec, d_model)
        返回: dx（对解码器输入的梯度）, denc_output（对编码器输出的梯度）
        """
        x, enc_output, self_attention_out, out1, cross_attention_out, out2, ff_out = (
            self.cache
        )

        # 第三个残差连接的反向传播
        dout2_1, dff_out = self.residual3.backward(dout)

        # 前馈网络的反向传播
        dout2_2 = self.feed_forward.backward(dff_out)

        # 合并梯度
        dout2 = dout2_1 + dout2_2

        # 第二个残差连接的反向传播
        dout1_1, dcross_attention_out = self.residual2.backward(dout2)

        # 交叉注意力的反向传播
        # Q来自解码器（out1），K和V来自编码器（enc_output）
        dQ_cross, dK_cross, dV_cross = self.cross_attention.backward(
            dcross_attention_out
        )
        dout1_2 = dQ_cross
        denc_output = dK_cross + dV_cross

        # 合并梯度
        dout1 = dout1_1 + dout1_2

        # 第一个残差连接的反向传播
        dx_1, dself_attention_out = self.residual1.backward(dout1)

        # 自注意力的反向传播
        # 因为 Q=K=V=x，所以梯度需要累加
        dQ_self, dK_self, dV_self = self.self_attention.backward(dself_attention_out)
        dx_2 = dQ_self + dK_self + dV_self

        # 合并梯度
        dx = dx_1 + dx_2

        return dx, denc_output
