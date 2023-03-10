import os
import time
from sys import stderr
import numpy as np
import math
import torch


splits_LOSO = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv', 'data_5.csv']
splits_LOUO = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_G.csv', 'data_H.csv', 'data_I.csv']
# splits_LOUO  = ['data_B.csv']
splits_LOUO_NP = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_H.csv', 'data_I.csv']

gestures_SU = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
gestures_NP = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
gestures_KT = ['G1', 'G11', 'G12', 'G13', 'G14', 'G15']

kin_mean = np.array(
    [3.96244727e-02, 4.15799272e-02, - 6.83070747e-03, 3.07358803e-01,
     - 1.69539955e-01, 7.29103208e-01, - 1.06836717e-06, - 1.01300281e-05,
     8.35814823e-05, 8.18588843e-04, - 5.57324968e-03, 8.22664589e-03,
     - 4.37194611e-01, 8.07797231e-02, 3.73276555e-02, - 5.15801785e-02,
     1.95393872e-01, 2.01552995e-01, - 7.53027529e-01, 3.76279481e-05,
     3.94373625e-05, 4.60702483e-05, - 1.00148135e-03, - 3.38861852e-03,
     6.38935734e-03, - 3.51171099e-01]
)

kin_std = np.array(
    [0.00916209, 0.01079738, 0.0123268,  0.82264756, 0.6305075,  1.80781916,
     0.01005,    0.0076605,  0.00972187, 0.61654319, 0.78448967, 0.77226779,
     0.54318369, 0.01404273, 0.01615757, 0.01661573, 0.70776227, 0.69825229,
     1.86339887, 0.01253722, 0.01421792, 0.01246655, 0.63354137, 0.61277592,
     0.75719052, 0.53728868]
)


def log(file, msg):
    """Log a message.

    :param file: File object to which the message will be written.
    :param msg:  Message to log (str).
    """
    print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
    file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-5

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]