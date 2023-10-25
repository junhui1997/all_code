import collections
import random
import numpy as np
from operator import itemgetter

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, input_seq, label):  # 将数据加入buffer
        self.buffer.append((input_seq, label))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        input_seq, label = zip(*transitions)
        return np.array(input_seq), np.array(label)

    def linear_sample(self, batch_size):
        idx = np.arange(batch_size)
        transitions = itemgetter(*idx)(self.buffer)
        input_seq, label = zip(*transitions)
        return np.array(input_seq), np.array(label)

    def log_sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        idx = np.arange(len(self.buffer))  # list(range(len(self.buffer)))
        log_probs = np.linspace(0, 5, len(self.buffer))
        # 将对数概率转换为真正的概率
        # a = self.buffer[1, 2]
        probs = np.exp(log_probs)
        probs /= probs.sum()  # 确保概率之和为1
        selected_idx = np.random.choice(idx, size=batch_size, replace=False, p=probs)
        transitions = itemgetter(*selected_idx)(self.buffer)
        input_seq, label = zip(*transitions)
        return np.array(input_seq), np.array(label)

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
