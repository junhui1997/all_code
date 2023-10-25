import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Dataset_neural_pd(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='A', scale=False, timeenc=0, freq='h', seasonal_patterns=None, args=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.args = args
        self.seed = args.seed
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.timeenc = timeenc

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        drill_folder = '../dataset/neural_pd/'
        df_raw = pd.read_pickle(drill_folder+'pd_train.pkl')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        #df_raw = df_raw[::2]
        if self.scale:
            df_raw.iloc[:, :] = self.scaler.fit_transform(df_raw.values)
        val_list = []
        label_list = []
        for i in range(self.seq_len, df_raw.shape[0]-self.pred_len):
            val = df_raw.iloc[i - self.seq_len:i, :self.args.enc_in].to_numpy().astype('float64')
            label = df_raw[['eq1', 'eq2']].iloc[i - self.label_len:i+self.pred_len, :].to_numpy().astype('float64') # label+pred #只选取了其中的eq1和eq2作为训练
            val_list.append(val)
            label_list.append(label)
        df_clean = pd.DataFrame({"value list": val_list, "label": label_list})
        split_mode = 'single'
        if split_mode == 'single':
            #df_clean = df_clean[::2]
            x_train, x_test_val = train_test_split(df_clean, test_size=0.2, random_state=self.seed)
            x_test, x_val = train_test_split(x_test_val, test_size=0.5, random_state=self.seed)
        # 不drop的话还会保存原本的index并形成新的一列
        x_train = x_train.reset_index(drop=True)
        x_val = x_val.reset_index(drop=True)
        x_test = df_clean.reset_index(drop=True)  # test all data
        #x_test = x_test.reset_index(drop=True)
        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        if self.flag == 'train':
            self.data_x = x_train
            self.ds_len = len(x_train)
        elif self.flag == 'val':
            self.data_x = x_val
            self.ds_len = len(x_val)
        elif self.flag == 'test':
            self.data_x = x_test
            self.ds_len = len(x_test)

        self.data_stamp_x = np.zeros((self.seq_len, 4), dtype=float)  # 时序数据这里要写成4，不知道为啥bone_dirll那个里面没事
        self.data_stamp_y = np.zeros((self.label_len+self.pred_len, 4), dtype=float)


    def __getitem__(self, index):
        # sssssss
        #     lllpppp
        # seq_x是输入进encoder的值，seq_y是输入进decoder的值
        seq_x = self.data_x['value list'].iloc[index].astype('float64')
        seq_y = self.data_x['label'].iloc[index].astype('float64')
        seq_x = torch.from_numpy(seq_x)
        seq_y = torch.from_numpy(seq_y)
        seq_x_mark = self.data_stamp_x
        seq_y_mark = self.data_stamp_y

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # 这里有个问题就是原本的数据是维度是enc_in,现在不是了，所以先给扩张一下，再给变换回去
        return self.scaler.inverse_transform(data)