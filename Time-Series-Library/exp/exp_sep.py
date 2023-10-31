from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_all, extract_seq
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from neural_pd.py_simu_eval import eval_simu
from neural_pd.gengerate_model import neural_pd
from neural_pd.pd_block import ReplayBuffer
from neural_pd.all_func import stiffness, coriolis, gravity, dynamics, disturbance, plot_all, apply_norm

warnings.filterwarnings('ignore')


class Exp_sep(Exp_Basic):
    def __init__(self, args):
        super(Exp_sep, self).__init__(args)

    def _build_model(self):
        # 从这里生成的模型，在这里做出改变即可
        model = neural_pd(self.model_dict[self.args.model].Model(self.args), self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion



    def train(self, setting):
        test_in = torch.rand((self.args.batch_size, self.args.seq_len, self.args.enc_in)).to(self.device)
        test_out = self.model(test_in)
        a = 1




        folder_path = './test_results/neural_pd/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        buffer_size = 2000  # 不能设置太大2000
        minimal_size = self.args.batch_size  # 64
        batch_size = self.args.batch_size
        replay_buffer = ReplayBuffer(buffer_size)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        self.model.train()

        # sys config
        counter = 0
        sample_rate = 0.0025
        total_t = 32
        total_epoch = int(total_t / sample_rate)
        dim = 2
        q_init = np.array([[-1], [0]])
        dq_init = np.array([[0], [0]])
        t1 = np.arange(0, 32.0025, 0.0025)
        t1 = t1.reshape(-1, 1)
        tout = t1
        q1_ref = np.sin(np.pi / 4 * tout) - np.pi / 2
        q2_ref = np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2))
        dq1_ref = np.pi / 4 * np.cos(np.pi / 4 * tout)
        dq2_ref = np.pi / 2 * -np.pi / 8 * np.sin(np.pi / 4 * (tout + np.pi / 2)) * np.sin(np.pi / 8 * (tout + np.pi / 2)) + np.pi / 2 * np.pi / 4 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.cos(
            np.pi / 4 * (tout + np.pi / 2))
        q_ref = np.concatenate((q1_ref, q2_ref), axis=1)
        dq_ref = np.concatenate((dq1_ref, dq2_ref), axis=1)
        total_eq = np.array([[], []])
        # total_edq = np.array([[], []])
        total_q = q_init
        total_dq = dq_init
        total_u = np.array([[0], [0]])
        e_predv = np.array([[0], [0]])

        # pd controller
        kp1 = 1000
        kp2 = 800
        kd1 = 500
        kd2 = 80
        kp = np.diag([kp1, kp2])
        kd = np.diag([kd1, kd2])
        while counter < total_epoch:
            if counter > self.args.seq_len + 1:
                if self.args.input_type == 'actual':
                    total_info = np.concatenate((total_q.transpose()[-self.args.seq_len - 1:-1, :], total_dq.transpose()[-self.args.seq_len - 1:-1, :], total_eq.transpose()[-self.args.seq_len - 1:-1, :]), axis=1)[:,
                                 :self.args.enc_in]
                else:
                    total_info = np.concatenate((q_ref[counter - self.args.seq_len:counter, :], dq_ref[counter - self.args.seq_len:counter, :], total_eq.transpose()[-self.args.seq_len - 1:-1, :]), axis=1)[:,
                                 :self.args.enc_in]
                # total_info = apply_norm(total_info)
                replay_buffer.add(total_info, total_eq.transpose()[-1, :])
            if replay_buffer.size() >= minimal_size:
                inputs, labels = replay_buffer.log_sample(batch_size)
                # inputs, labels = replay_buffer.linear_sample(batch_size) # 使用linear sample时候buffer size == batch_size
                # inputs, labels = replay_buffer.sample(batch_size)
                inputs = torch.from_numpy(inputs).to(torch.float32).to(self.device)  # .requires_grad_(True)
                labels = torch.from_numpy(labels).to(torch.float32).to(self.device)
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                loss.backward()
                optimizer.step()
                # print(inputs.grad)
                # predict current error
                self.model.eval()
                if self.args.input_type == 'actual':
                    current_inputs = np.concatenate((total_q.transpose()[-self.args.seq_len:, :], total_dq.transpose()[-self.args.seq_len:, :], total_eq.transpose()[-self.args.seq_len:, :]), axis=1)[:, :self.args.enc_in]
                else:
                    current_inputs = np.concatenate((q_ref[counter - self.args.seq_len + 1:counter + 1, :], dq_ref[counter - self.args.seq_len + 1:counter + 1, :], total_eq.transpose()[-self.args.seq_len:, :]),
                                                    axis=1)[:, :self.args.enc_in]
                # current_inputs = apply_norm(current_inputs)
                current_inputs = torch.from_numpy(current_inputs).to(torch.float32).to(self.device)
                with torch.no_grad():
                    e_pred = self.model(current_inputs.unsqueeze(0)).cpu().detach().numpy().transpose()
                print(e_pred)
                # de_pred = (e_pred-e_predv)/sample_rate
                de_pred = np.array([[0], [0]])
                e_predv = e_pred
            else:
                e_pred = np.array([[0], [0]])
                de_pred = np.array([[0], [0]])
            e_q_t = q_init - q_ref[counter].reshape(dim, 1)
            e_q = e_q_t + e_pred
            e_dq = dq_init - dq_ref[counter].reshape(dim, 1) + de_pred
            total_eq = np.concatenate((total_eq, e_q_t), axis=1)
            u_pd = -kp @ e_q - kd @ e_dq
            u_pd = np.clip(u_pd, -130, 130)
            M = stiffness(q_init)
            C = coriolis(q_init, dq_init)
            G = gravity(q_init)
            disturb = disturbance(counter * sample_rate)
            TL = u_pd - C - G + disturb
            ddq = dynamics(TL, M)
            dq = dq_init + ddq * sample_rate
            q = q_init + dq * sample_rate
            dq_init = dq
            q_init = q
            counter += 1
            total_dq = np.concatenate((total_dq, dq), axis=1)
            total_q = np.concatenate((total_q, q), axis=1)
            total_u = np.concatenate((total_u, u_pd), axis=1)
            print(counter)
            # 这样相当于是有重力但是没有disturbance

        q = total_q.transpose()
        dq = total_dq.transpose()
        sTau = total_u.transpose()
        tout = np.linspace(0, 32, num=len(t1))  # 不使用np速度回很慢
        plot_all(q, dq, sTau, tout, folder_path)



