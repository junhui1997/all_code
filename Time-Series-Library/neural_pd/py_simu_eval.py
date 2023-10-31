import numpy as np
from neural_pd.all_func import stiffness, coriolis, gravity, dynamics, disturbance, plot_all, plot_compare
from neural_pd.pd_block import ReplayBuffer
import torch



def eval_simu(model, args, folder_path, to_plot=None):
    # sys config
    dim = args.c_out
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    counter = 0
    sample_rate = 0.0025
    total_t = 32
    total_epoch = int(total_t / sample_rate)
    q_init = np.array([[-1], [0]])
    dq_init = np.array([[0], [0]])
    t1 = np.arange(0, 32.0025, 0.0025)
    t1 = t1.reshape(-1, 1)
    tout = t1
    q1_ref = np.sin(np.pi / 4 * tout) - np.pi / 2
    q2_ref = np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2))
    dq1_ref = np.pi / 4 * np.cos(np.pi / 4 * tout)
    dq2_ref = np.pi / 2 * -np.pi / 8 * np.sin(np.pi / 4 * (tout + np.pi / 2)) * np.sin(
        np.pi / 8 * (tout + np.pi / 2)) + np.pi / 2 * np.pi / 4 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.cos(
        np.pi / 4 * (tout + np.pi / 2))
    q_ref = np.concatenate((q1_ref, q2_ref), axis=1)
    dq_ref = np.concatenate((dq1_ref, dq2_ref), axis=1)
    total_eq = np.array([[], []])
    total_eq_pred = np.array([[], []])
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
        if counter > args.seq_len + 1:
            # 限定条件 seq_len>=label_len>=pred_len
            model.eval()
            #current_inputs = np.concatenate((total_q.transpose()[-args.seq_len:, :], total_dq.transpose()[-args.seq_len:, :], total_eq.transpose()[-args.seq_len:, :]), axis=1)
            current_inputs = np.concatenate((q_ref[counter - args.seq_len + 1:counter + 1, :], dq_ref[counter - args.seq_len + 1:counter + 1, :], total_eq.transpose()[-args.seq_len:, :]), axis=1)[:, :args.enc_in]
            current_inputs = torch.from_numpy(current_inputs).float().to(device).unsqueeze(0)  # (1,seq_len,enc_in)
            batch_x = current_inputs
            batch_y = current_inputs[:, -args.label_len:, 4:6]

            batch_x_mark = torch.zeros(1, args.seq_len, 4).float().to(device)
            batch_y_mark = torch.zeros(1, args.label_len+args.pred_len, 4).float().to(device)

            # decoder input
            # batch_y原本是label+pred,现在将pred部分全部替换为0
            # 对于timenet模型没有使用decoder部分，相当于是直接从encoder部分给映射出去了
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()  # (1,pred_len,enc_in)
            dec_inp = torch.cat([batch_y, dec_inp], dim=1).float().to(device)  # (1,label_len + pred_len,enc_in)

            # # pred_len >1
            # with torch.no_grad():
            #     outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # (1,pred_len,c_out)
            #     e_pred0 = outputs[:, 0, :].cpu().detach().numpy().transpose()   # (c_out,1)
            #     outputs = torch.cumsum(outputs, dim=1) # 沿着pred_len维度（即维度1）累加
            #     e_pred = outputs[:, 2, :].cpu().detach().numpy().transpose()
            #     # print(e_pred)
            # de_pred = (e_pred - e_predv) / sample_rate
            # de_pred = np.array([[0], [0]])
            # e_predv = e_pred

            # pred_len 1
            with torch.no_grad():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # (1,pred_len,c_out)
                e_pred = outputs[:, 0, :].cpu().detach().numpy().transpose()   # (c_out,1)

            de_pred = (e_pred - e_predv) / sample_rate
            e_predv = e_pred
            total_eq_pred = np.concatenate((total_eq_pred, e_pred), axis=1)

        else:
            e_pred = np.array([[0], [0]])
            de_pred = np.array([[0], [0]])
            total_eq_pred = np.concatenate((total_eq_pred, e_pred), axis=1)
        e_q_t = q_init - q_ref[counter].reshape(dim, 1)
        e_q = e_q_t + e_pred
        e_dq = dq_init - dq_ref[counter].reshape(dim, 1) + de_pred
        total_eq = np.concatenate((total_eq, e_q_t), axis=1)
        u_pd = -kp @ e_q - kd @ e_dq
        #u_pd = np.clip(u_pd, -130, 130)
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

    q = total_q.transpose()
    dq = total_dq.transpose()
    sTau = total_u.transpose()
    tout = np.linspace(0, 32, num=len(t1))  # 不使用np速度回很慢
    if True:
        plot_compare(total_eq.transpose(), total_eq_pred.transpose(), 'eq', 'eq_pred', folder_path, 'e&eq')

    if to_plot is not None:
        q = to_plot+q_ref
    plot_all(q, dq, sTau, tout, folder_path)

