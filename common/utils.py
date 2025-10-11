# coding: utf-8
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

def clip_grads(grads, max_norm):
  total_norm = 0
  for grad in grads:
    total_norm += np.sum(grad ** 2)
  total_norm = np.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)
  if rate < 1:
    for grad in grads:
      grad *= rate

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
  print('evaluating perplexity ...')
  corpus_size = len(corpus)
  total_loss = 0
  max_iters = (corpus_size - 1) // (batch_size * time_size)
  jump = (corpus_size - 1) // batch_size

  for iters in range(max_iters):
    xs = np.zeros((batch_size, time_size), dtype=np.int32)
    ts = np.zeros((batch_size, time_size), dtype=np.int32)
    time_offset = iters * time_size
    offsets = [time_offset + (i * jump) for i in range(batch_size)]
    for t in range(time_size):
      for i, offset in enumerate(offsets):
        xs[i, t] = corpus[(offset + t) % corpus_size]
        ts[i, t] = corpus[(offset + t + 1) % corpus_size]

    try:
      loss = model.forward(xs, ts, train_flg=False)
    except TypeError:
      loss = model.forward(xs, ts)
    total_loss += loss

    sys.stdout.write('\r%d / %d' % (iters, max_iters))
    sys.stdout.flush()

  print('')
  ppl = np.exp(total_loss / max_iters)
  return ppl

class Trainer:
  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer = optimizer
    self.loss_list = []
    self.eval_interval = None
    self.current_epoch = 0

  def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
    data_size = len(x)
    max_iters = data_size // batch_size
    self.eval_interval = eval_interval
    model, optimizer = self.model, self.optimizer
    total_loss = 0
    loss_count = 0

    start_time = time.time()
    for epoch in range(max_epoch):
      idx = np.random.permutation(np.arange(data_size))
      x = x[idx]
      t = t[idx]

      for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
        if max_grad is not None:
          clip_grads(grads, max_grad)
        optimizer.update(params, grads)
        total_loss += loss
        loss_count += 1

        if (eval_interval is not None) and (iters % eval_interval) == 0:
          avg_loss = total_loss / loss_count
          elapsed_time = time.time() - start_time
          print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f' % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
          self.loss_list.append(float(avg_loss))
          total_loss, loss_count = 0, 0

      self.current_epoch += 1

  def plot(self, ylim=None):
    x = np.arange(len(self.loss_list))
    if ylim is not None:
      plt.ylim(*ylim)
    plt.plot(x, self.loss_list, label='train')
    plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
    plt.ylabel('loss')
    plt.show()

class RnnlmTrainer:
  def __init__(self, model, optimizer):
      self.model = model
      self.optimizer = optimizer
      self.time_idx = None
      self.ppl_list = None
      self.eval_interval = None
      self.current_epoch = 0

  def get_batch(self, x, t, batch_size, time_size):
      batch_x = np.empty((batch_size, time_size), dtype='i')
      batch_t = np.empty((batch_size, time_size), dtype='i')

      data_size = len(x)
      jump = data_size // batch_size
      offsets = [i * jump for i in range(batch_size)]

      for time_ in range(time_size):
        for i, offset in enumerate(offsets):
          batch_x[i, time_] = x[(offset + self.time_idx) % data_size]
          batch_t[i, time_] = t[(offset + self.time_idx) % data_size]
        self.time_idx += 1
      return batch_x, batch_t

  def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
    data_size = len(xs)
    max_iters = data_size // (batch_size * time_size)
    self.time_idx = 0
    self.ppl_list = []
    self.eval_interval = eval_interval
    model, optimizer = self.model, self.optimizer
    total_loss = 0
    loss_count = 0

    start_time = time.time()
    for epoch in range(max_epoch):
      for iters in range(max_iters):
        batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

        loss = model.forward(batch_x, batch_t)
        model.backward()
        params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
        if max_grad is not None:
          clip_grads(grads, max_grad)
        optimizer.update(params, grads)
        total_loss += loss
        loss_count += 1

        if (eval_interval is not None) and (iters % eval_interval) == 0:
          ppl = np.exp(total_loss / loss_count)
          elapsed_time = time.time() - start_time
          print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f' % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
          self.ppl_list.append(float(ppl))
          total_loss, loss_count = 0, 0

        self.current_epoch += 1

  def plot(self, ylim=None):
    x = np.arange(len(self.ppl_list))
    if ylim is not None:
      plt.ylim(*ylim)
    plt.plot(x, self.ppl_list, label='train')
    plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
    plt.ylabel('perplexity')
    plt.show()

def remove_duplicate(params, grads):
  params, grads = params[:], grads[:]

  while True:
    find_flg = False
    L = len(params)

    for i in range(0, L - 1):
      for j in range(i + 1, L):
        if params[i] is params[j]:
          grads[i] += grads[j]
          find_flg = True
          params.pop(j)
          grads.pop(j)
        elif params[i].ndim == 2 and params[j].ndim == 2 and \
          params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
          grads[i] += grads[j].T
          find_flg = True
          params.pop(j)
          grads.pop(j)

        if find_flg:
          break
      if find_flg:
        break

    if not find_flg:
      break

  return params, grads

def eval_seq2seq(model, question, correct, id_to_char, verbose=False, is_reverse=False):
  correct = correct.flatten()
  
  start_id = correct[0]
  correct = correct[1:]
  guess = model.generate(question, start_id, len(correct))

  question = ''.join([id_to_char[int(c)] for c in question.flatten()])
  correct = ''.join([id_to_char[int(c)] for c in correct])
  guess = ''.join([id_to_char[int(c)] for c in guess])

  if verbose:
    if is_reverse:
      question = question[::-1]

    colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
    print('Q', question)
    print('T', correct)

    is_windows = os.name == 'nt'

    if correct == guess:
      mark = colors['ok'] + '☑' + colors['close']
      if is_windows:
        mark = 'O'
      print(mark + ' ' + guess)
    else:
      mark = colors['fail'] + '☒' + colors['close']
      if is_windows:
        mark = 'X'
      print(mark + ' ' + guess)
    print('---')

  return 1 if guess == correct else 0
