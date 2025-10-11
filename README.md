# nlp-demo

 本项目用于学习并复现《Deep Learning from Scratch 2》中的经典 NLP 序列建模算法，包含 RNN 语言模型、LSTM 语言模型、Seq2seq（含 Peeky 解码器）、以及 Attention-Seq2seq。实现基于 NumPy 与最小依赖。

- 原项目链接: https://github.com/oreilly-japan/deep-learning-from-scratch-2
- Python 版本: 3.11（见 `.python-version`、`pyproject.toml`）
- 依赖: `numpy`, `matplotlib`, `ipykernel(dev)`（见 `pyproject.toml`）

## 项目结构

- 代码
  - `rnn/`、`lstm/`、`seq2seq/`、`attention/`、`common/`、`dataset/`
- 数据与可视化
  - `dataset/data/`：PTB 文本与字符级任务数据 `addition.txt`、`date.txt`
  - 结果图：`rnn_perplexity.png`、`seq2seq_val_acc.png`、`attention_seq2seq_val_acc.png`
- Notebook
  - `rnn_main.ipynb`、`lstm_main.ipynb`、`seq2seq_main.ipynb`、`attention_main.ipynb`

## 复现范围

- RNN 语言模型
  - 模型: `rnn/simple_rnnlm.py`
  - 时序层: `rnn/rnn_layers.py` 中的 `RNN`、`TimeRNN`
- LSTM 语言模型（含 Dropout、权重共享）
  - 模型: `lstm/gated_rnnlm.py`
  - 时序层: `lstm/lstm_layers.py` 中的 `LSTM`、`TimeLSTM`
- Seq2seq（Peeky 解码器）
  - 顶层: `seq2seq/seq2seq.py`
  - 子层: `seq2seq/seq2seq_layers.py`（`Encoder`、`Decoder`，将编码器隐藏状态拼接至输入与仿射层输入）
- Attention-Seq2seq
  - 顶层: `attention/attention_seq2seq.py`
  - 子层: `attention/attention_layers.py`（`AttentionWeight`、`WeightSum`、`TimeAttention`）
- 公共模块
  - `common/layers.py`（`TimeEmbedding`、`TimeAffine`、`TimeSoftmaxWithLoss`、`TimeDropout` 等）
  - `common/optimizer.py`（`SGD`、`Adam`）
  - `common/utils.py`（`Trainer`、`RnnlmTrainer`、梯度裁剪、困惑度评估、Seq2seq 评估等）
  - `common/functions.py`（激活函数、softmax、交叉熵）

## 环境与安装（uv）

- 本项目使用 uv 进行包管理。安装 uv 后，在项目根目录运行：

```bash
 uv sync
```

 这将基于 `pyproject.toml` 同步并创建虚拟环境及依赖。

## 数据集

- PTB（Penn Treebank）词级语言建模：`dataset/ptb.py`（首次会下载到 `dataset/data/`，仓库已包含缓存文件）
- 字符级序列任务：`dataset/data/addition.txt`、`dataset/data/date.txt`，加载工具见 `dataset/sequence.py`

## 快速开始

 打开根目录下的 `*_main.ipynb` 并按节执行，即可复现实验与图表。

## 参考与致谢

- 书籍与代码：[oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
- 本项目仅用于学习与教学目的，欢迎对实现细节提出改进建议。
