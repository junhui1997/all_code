import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy

def stiffness(q):
    q1, q2 = q[0, 0], q[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.81

    M11 = m1 * l1 ** 2 / 3 + m2 * l1 ** 2 + m2 * l2 ** 2 / 4 + m2 * l1 * l2 * c2
    M12 = m2 * l2 ** 2 / 4 + m2 * l1 * l2 * c2 / 2
    M22 = m2 * l2 ** 2 / 3

    M = np.array([[M11], [M12],
                  [M12], [M22]])

    return M


def coriolis(q, dq):
    q1, q2 = q[0, 0], q[1, 0]
    dq1, dq2 = dq[0, 0], dq[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.80665

    C11 = -m2 * l1 * l2 * s2 * dq2
    C12 = -m2 * l1 * l2 * s2 * dq2 / 2
    C21 = m2 * l1 * l2 * s2 * dq1 / 2
    C22 = 0

    CM = np.array([[C11, C12],
                   [C21, C22]])

    C = np.dot(CM, dq)

    return C


def gravity(q):
    q1, q2 = q[0, 0], q[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)
    c12 = np.cos(q1 + q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.80665

    G1 = m1 * g * l1 * c1 / 2 + m2 * g * l1 * c1 + m2 * g * l2 * c12 / 2
    G2 = m2 * g * l2 * c12 / 2

    G = np.array([[G1], [G2]])

    return G

def dynamics(TL, M):
    M = M.reshape(-1)
    #print(M.shape)
    M22 = np.array([[M[0], M[1]],
                    [M[2], M[3]]])
    #print(M22.shape)
    ddq = np.linalg.solve(M22, TL)

    return ddq

def disturbance(t):
    if t > 10:
        d = 50.0 * np.array([np.sin(0.6 * (t - 10)), np.sin(0.6 * (t - 10))])
    else:
        d = np.zeros(2)
    return d.reshape(2, 1)

def plot_all(q, dq, sTau, tout, folder_path):

    # load baseline data
    mat_g = scipy.io.loadmat('neural_pd/gravity.mat')
    mat_og = scipy.io.loadmat('neural_pd/gravity_offset.mat')
    q_g = mat_g['q']
    dq_g = mat_g['dq']
    e1_g = mat_g['e1']
    sTau_g = mat_g['sTau']
    q_og = mat_og['q']
    dq_og = mat_og['dq']
    e1_og = mat_og['e1']
    sTau_og = mat_og['sTau']

    counter = 0
    plt.figure(1)
    plt.subplot(321)
    plt.plot(tout, np.sin(np.pi / 4 * tout) - np.pi / 2, linewidth=0.5, label='gt')
    plt.plot(tout, q[:, 0], linewidth=0.5, label='ours')
    plt.plot(tout, q_g[:, 0], linewidth=0.5, label='g')
    plt.plot(tout, q_og[:, 0], linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$q_1$ (rad)')

    plt.subplot(323)
    plt.plot(tout, abs(np.sin(np.pi / 4 * tout) - np.pi / 2 - q[:, 0]) * 180 / np.pi,  linewidth=0.5, label='our')
    plt.plot(tout, abs(np.sin(np.pi / 4 * tout) - np.pi / 2 - q_g[:, 0]) * 180 / np.pi,  linewidth=0.5, label='g')
    plt.plot(tout, abs(np.sin(np.pi / 4 * tout) - np.pi / 2 - q_og[:, 0]) * 180 / np.pi,  linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e_{q_1}$ (deg)')

    plt.subplot(322)
    plt.plot(tout, np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)), linewidth=0.5, label='gt')
    plt.plot(tout, q[:, 1], linewidth=0.5, label='our')
    plt.plot(tout, q_g[:, 1], linewidth=0.5, label='g')
    plt.plot(tout, q_og[:, 1], linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$q_2$ (rad)')

    plt.subplot(324)
    plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q[:, 1]) * 180 / np.pi, linewidth=0.5, label='our')
    plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q_g[:, 1]) * 180 / np.pi, linewidth=0.5, label='g')
    plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q_og[:, 1]) * 180 / np.pi, linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e_{q_2}$ (deg)')

    plt.subplot(325)
    plt.plot(tout, sTau[:, 0],  linewidth=0.5, label='our')
    plt.plot(tout, sTau_g[:, 0], linewidth=0.5, label='g')
    plt.plot(tout, sTau_og[:, 0],  linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\tau_1$ [Nm]')

    plt.subplot(326)
    plt.plot(tout, sTau[:, 1], linewidth=0.5, label='our')
    plt.plot(tout, sTau_g[:, 1],  linewidth=0.5, label='g')
    plt.plot(tout, sTau_og[:, 1],  linewidth=0.5, label='og')
    plt.ylabel(r'$\tau_2$ [Nm]')
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(folder_path + 'pos_{}.svg'.format(counter), format='svg')



    # plot the second graph
    plt.figure(2)
    plt.subplot(221)
    t = tout
    pi = np.pi
    plt.plot(tout, pi / 4 * np.cos(pi / 4 * tout), linewidth=0.5, label='gt')
    plt.plot(tout, dq[:, 0], linewidth=0.5, label='our')
    plt.plot(tout, dq_g[:, 0], linewidth=0.5, label='g')
    plt.plot(tout, dq_og[:, 0], linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\dot{q}_1$ (rad/s)')

    plt.subplot(223)
    plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq[:, 0]), linewidth=0.5, label='our')
    plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq_g[:, 0]), linewidth=0.5, label='g')
    plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq_og[:, 0]), linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e\dot{q}_1$ (rad/s)')

    plt.subplot(222)
    plt.plot(tout, pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
        pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
        pi / 4 * (t + np.pi / 2)), linewidth=0.5, label='gt')
    plt.plot(tout, dq[:, 1], linewidth=0.5, label='our')
    plt.plot(tout, dq_g[:, 1], linewidth=0.5, label='g')
    plt.plot(tout, dq_og[:, 1], linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\dot{q}_2$ (rad/s)')

    plt.subplot(224)
    plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
        pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
        pi / 4 * (t + np.pi / 2)) - dq[:, 1]), linewidth=0.5, label='our')
    plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
        pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
        pi / 4 * (t + np.pi / 2)) - dq_g[:, 1]), linewidth=0.5, label='g')
    plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
        pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
        pi / 4 * (t + np.pi / 2)) - dq_og[:, 1]), linewidth=0.5, label='og')
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e\dot{q}_2$ (rad/s)')
    plt.savefig(folder_path + 'vel_{}.svg'.format(counter), format='svg')
    plt.show()


def arrays_to_dataframe(total_q, total_dq, total_eq, total_edq):
    """
    Convert four numpy arrays with shape (12800, 2) into a DataFrame with eight columns.

    Parameters:
    - total_q, total_dq, total_eq, total_edq: numpy.array of shape (12800, 2)

    Returns:
    - DataFrame with columns 'q1', 'q2', 'dq1', 'dq2', 'eq1', 'eq2', 'edq1', 'edq2'
    """

    # Split each numpy.array into separate columns
    q1, q2 = total_q[:, 0], total_q[:, 1]
    dq1, dq2 = total_dq[:, 0], total_dq[:, 1]
    eq1, eq2 = total_eq[:, 0], total_eq[:, 1]
    edq1, edq2 = total_edq[:, 0], total_edq[:, 1]

    # Create and return a DataFrame
    return pd.DataFrame({
        'q1': q1,
        'q2': q2,
        'dq1': dq1,
        'dq2': dq2,
        'eq1': eq1,
        'eq2': eq2,
        'edq1': edq1,
        'edq2': edq2
    })