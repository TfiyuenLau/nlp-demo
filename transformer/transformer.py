# coding: utf-8
import numpy as np
from transformer.transformer_layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from common.layers import TimeEmbedding, PositionalEncoding


class TransformerEncoder:
    """Transformer编码器

    由多个相同的编码器层堆叠而成，每层包含自注意力和前馈网络。

    结构:
      输入序列 -> Embedding -> 位置编码 ->
        -> [EncoderLayer] x N -> 输出

    参数:
      vocab_size: 词汇表大小
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
      d_ff: 前馈网络的隐藏层维度
      num_layers: 编码器层的数量
      max_len: 最大序列长度
      dropout_ratio: dropout比率
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout_ratio=0.1,
    ):
        self.d_model = d_model
        self.num_layers = num_layers

        rn = np.random.randn

        # 词嵌入层
        embed_W = (rn(vocab_size, d_model) / 100).astype("f")
        self.embedding = TimeEmbedding(embed_W)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠多个编码器层
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_ratio)
            self.layers.append(layer)

        # 收集所有参数和梯度
        self.params = []
        self.grads = []

        # 添加embedding的参数
        self.params += self.embedding.params
        self.grads += self.embedding.grads

        # 位置编码没有可学习参数（是固定的）

        # 添加所有编码器层的参数
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, mask=None):
        """
        前向传播

        xs: 输入序列（词ID），形状 (N, T)
        mask: 可选的padding掩码，形状 (N, T, T)

        返回: 编码器输出，形状 (N, T, d_model)
        """
        # 词嵌入: (N, T) -> (N, T, d_model)
        out = self.embedding.forward(xs)

        # 缩放词嵌入（Transformer论文中的做法）
        out = out * np.sqrt(self.d_model)

        # 添加位置编码
        out = self.pos_encoding.forward(out)

        # 通过所有编码器层
        for layer in self.layers:
            out = layer.forward(out, mask)

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T, d_model)
        返回: None（输入是离散的词ID，没有梯度）
        """
        # 反向传播通过所有编码器层
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # 位置编码的反向传播
        dout = self.pos_encoding.backward(dout)

        # 词嵌入缩放的反向传播
        dout = dout * np.sqrt(self.d_model)

        # 词嵌入的反向传播
        self.embedding.backward(dout)

        return None


class TransformerDecoder:
    """Transformer解码器

    由多个相同的解码器层堆叠而成，每层包含自注意力、交叉注意力和前馈网络。

    结构:
      输入序列 -> Embedding -> 位置编码 ->
        -> [DecoderLayer with encoder output] x N -> 输出

    参数:
      vocab_size: 词汇表大小
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
      d_ff: 前馈网络的隐藏层维度
      num_layers: 解码器层的数量
      max_len: 最大序列长度
      dropout_ratio: dropout比率
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout_ratio=0.1,
    ):
        self.d_model = d_model
        self.num_layers = num_layers

        rn = np.random.randn

        # 词嵌入层
        embed_W = (rn(vocab_size, d_model) / 100).astype("f")
        self.embedding = TimeEmbedding(embed_W)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠多个解码器层
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_ratio)
            self.layers.append(layer)

        # 收集所有参数和梯度
        self.params = []
        self.grads = []

        # 添加embedding的参数
        self.params += self.embedding.params
        self.grads += self.embedding.grads

        # 位置编码没有可学习参数（是固定的）

        # 添加所有解码器层的参数
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_output, self_mask=None, cross_mask=None):
        """
        前向传播

        xs: 输入序列（词ID），形状 (N, T_dec)
        enc_output: 编码器输出，形状 (N, T_enc, d_model)
        self_mask: 自注意力掩码（因果掩码），形状 (N, T_dec, T_dec)
        cross_mask: 交叉注意力掩码（padding掩码），形状 (N, T_dec, T_enc)

        返回: 解码器输出，形状 (N, T_dec, d_model)
        """
        # 词嵌入: (N, T_dec) -> (N, T_dec, d_model)
        out = self.embedding.forward(xs)

        # 缩放词嵌入（Transformer论文中的做法）
        out = out * np.sqrt(self.d_model)

        # 添加位置编码
        out = self.pos_encoding.forward(out)

        # 通过所有解码器层
        for layer in self.layers:
            out = layer.forward(out, enc_output, self_mask, cross_mask)

        return out

    def backward(self, dout):
        """
        反向传播

        dout: 上游梯度，形状 (N, T_dec, d_model)
        返回: denc_output（对编码器输出的梯度累积）
        """
        # 累积对编码器输出的梯度
        denc_output = 0

        # 反向传播通过所有解码器层
        for layer in reversed(self.layers):
            dout, denc = layer.backward(dout)
            denc_output = denc_output + denc

        # 位置编码的反向传播
        dout = self.pos_encoding.backward(dout)

        # 词嵌入缩放的反向传播
        dout = dout * np.sqrt(self.d_model)

        # 词嵌入的反向传播
        self.embedding.backward(dout)

        return denc_output

    def generate(self, enc_output, start_id, max_len, cross_mask=None):
        """
        自回归生成序列（用于推理）

        enc_output: 编码器输出，形状 (N, T_enc, d_model)
        start_id: 起始token的ID
        max_len: 生成的最大长度
        cross_mask: 交叉注意力掩码

        返回: 生成的词ID序列，列表形式
        """
        N = enc_output.shape[0]
        assert N == 1, "生成时batch size必须为1"

        # 初始化输出序列
        generated = [start_id]

        for _ in range(max_len):
            # 当前序列: (1, current_len)
            xs = np.array(generated, dtype=np.int32).reshape(1, -1)

            # 创建因果掩码（下三角矩阵）
            current_len = xs.shape[1]
            self_mask = self._create_causal_mask(N, current_len)

            # 前向传播
            out = self.embedding.forward(xs)
            out = out * np.sqrt(self.d_model)
            out = self.pos_encoding.forward(out)

            for layer in self.layers:
                out = layer.forward(out, enc_output, self_mask, cross_mask)

            # 取最后一个位置的输出
            # out: (1, current_len, d_model)
            last_out = out[:, -1, :]  # (1, d_model)

            # 注意：这里返回的是特征向量，需要外部的输出层将其转换为词概率
            # 在完整模型中会有一个线性层 + softmax
            # 这里我们只返回特征，由TransformerSeq2seq处理
            return last_out, generated

        return None, generated

    def _create_causal_mask(self, batch_size, seq_len):
        """
        创建因果掩码（防止看到未来的信息）

        返回下三角矩阵，上三角部分为True（会被mask掉）

        例如 seq_len=4:
          [[False, True,  True,  True ],
           [False, False, True,  True ],
           [False, False, False, True ],
           [False, False, False, False]]
        """
        # 创建上三角矩阵（不包括对角线）
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        # 扩展batch维度
        mask = np.tile(mask, (batch_size, 1, 1))
        return mask


class TransformerSeq2seq:
    """完整的Transformer序列到序列模型

    结合编码器和解码器，用于序列到序列的转换任务（如机器翻译、日期转换等）。

    结构:
      输入序列 -> Encoder -> 编码器输出 -> Decoder -> 输出层 -> 预测

    参数:
      vocab_size: 词汇表大小（源语言和目标语言使用相同词汇表）
      d_model: 模型的特征维度
      num_heads: 注意力头的数量
      d_ff: 前馈网络的隐藏层维度
      num_layers: 编码器和解码器的层数
      max_len: 最大序列长度
      dropout_ratio: dropout比率
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout_ratio=0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 编码器
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout_ratio
        )

        # 解码器
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout_ratio
        )

        # 输出层：将解码器的输出映射到词汇表
        # d_model -> vocab_size
        rn = np.random.randn
        self.output_W = (rn(d_model, vocab_size) / np.sqrt(d_model)).astype("f")
        self.output_b = np.zeros(vocab_size, dtype="f")

        # 损失层
        from common.layers import TimeSoftmaxWithLoss

        self.loss_layer = TimeSoftmaxWithLoss()

        # 收集所有参数和梯度
        self.params = self.encoder.params + self.decoder.params
        self.params += [self.output_W, self.output_b]

        self.grads = self.encoder.grads + self.decoder.grads
        self.grads += [np.zeros_like(self.output_W), np.zeros_like(self.output_b)]

        # 缓存用于反向传播
        self.cache = None

    def forward(self, xs, ts):
        """
        前向传播（训练时）

        xs: 源序列（输入），形状 (N, T_src)
        ts: 目标序列（标签），形状 (N, T_tgt)

        返回: 损失值（标量）
        """
        # 解码器的输入和标签（teacher forcing）
        # 输入是目标序列去掉最后一个token
        # 标签是目标序列去掉第一个token
        decoder_xs = ts[:, :-1]
        decoder_ts = ts[:, 1:]

        N, T_src = xs.shape
        N, T_tgt = decoder_xs.shape

        # 创建掩码
        # 编码器的padding掩码（可选，这里简化处理，不使用）
        enc_mask = None

        # 解码器的因果掩码（防止看到未来）
        dec_self_mask = self._create_causal_mask(N, T_tgt)

        # 交叉注意力的padding掩码（可选，这里简化处理，不使用）
        dec_cross_mask = None

        # 编码器前向传播
        enc_output = self.encoder.forward(xs, enc_mask)

        # 解码器前向传播
        dec_output = self.decoder.forward(
            decoder_xs, enc_output, dec_self_mask, dec_cross_mask
        )

        # 输出层：(N, T_tgt, d_model) -> (N, T_tgt, vocab_size)
        N, T, D = dec_output.shape
        dec_output_2d = dec_output.reshape(N * T, D)
        score_2d = np.dot(dec_output_2d, self.output_W) + self.output_b
        score = score_2d.reshape(N, T, self.vocab_size)

        # 缓存用于反向传播
        self.cache = dec_output

        # 计算损失
        loss = self.loss_layer.forward(score, decoder_ts)

        return loss

    def backward(self, dout=1):
        """
        反向传播

        dout: 上游梯度（默认为1）
        返回: None
        """
        # 损失层的反向传播
        dscore = self.loss_layer.backward(dout)

        N, T, V = dscore.shape

        # 输出层的反向传播
        dscore_2d = dscore.reshape(N * T, V)
        ddec_output_2d = np.dot(dscore_2d, self.output_W.T)
        ddec_output = ddec_output_2d.reshape(N, T, self.d_model)

        # 计算输出层参数的梯度
        dec_output = self.cache  # 从forward中获取缓存的dec_output
        dec_output_2d = dec_output.reshape(N * T, self.d_model)
        dW_output = np.dot(dec_output_2d.T, dscore_2d)
        db_output = np.sum(dscore_2d, axis=0)

        self.grads[-2][...] = dW_output
        self.grads[-1][...] = db_output

        # 解码器的反向传播
        denc_output = self.decoder.backward(ddec_output)

        # 编码器的反向传播
        self.encoder.backward(denc_output)

        return None

    def generate(self, xs, start_id, max_len):
        """
        生成序列（推理时）

        xs: 源序列，形状 (1, T_src)（batch size必须为1）
        start_id: 起始token的ID
        max_len: 生成的最大长度

        返回: 生成的词ID列表
        """
        N = xs.shape[0]
        assert N == 1, "生成时batch size必须为1"

        # 编码器前向传播
        enc_output = self.encoder.forward(xs, mask=None)

        # 初始化生成序列
        generated = [start_id]

        # 自回归生成
        for _ in range(max_len):
            # 当前解码器输入
            decoder_xs = np.array(generated, dtype=np.int32).reshape(1, -1)
            T_dec = decoder_xs.shape[1]

            # 创建因果掩码
            dec_self_mask = self._create_causal_mask(N, T_dec)

            # 解码器前向传播
            dec_output = self.decoder.forward(
                decoder_xs, enc_output, dec_self_mask, None
            )

            # 取最后一个位置的输出
            last_output = dec_output[:, -1, :]  # (1, d_model)

            # 通过输出层得到词概率
            score = (
                np.dot(last_output, self.output_W) + self.output_b
            )  # (1, vocab_size)

            # 选择概率最高的词
            next_id = np.argmax(score.flatten())
            generated.append(int(next_id))

            # 如果生成了结束符，可以提前停止（这里简化处理）
            # if next_id == end_id:
            #     break

        # 返回生成的序列，去掉第一个start_id（与AttentionSeq2seq保持一致）
        return generated[1:]

    def _create_causal_mask(self, batch_size, seq_len):
        """
        创建因果掩码（防止看到未来的信息）

        返回下三角矩阵，上三角部分为True（会被mask掉）
        """
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        mask = np.tile(mask, (batch_size, 1, 1))
        return mask

    def save_params(self, file_name="TransformerSeq2seq.pkl"):
        """保存模型参数"""
        import pickle

        params = [p.astype(np.float16) for p in self.params]
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="TransformerSeq2seq.pkl"):
        """加载模型参数"""
        import pickle
        import os

        if "/" in file_name:
            file_name = file_name.replace("/", os.sep)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [p.astype("f") for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
