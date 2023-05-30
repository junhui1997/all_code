import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from data_provider.uea import subsample, interpolate_missing, Normalizer

# 设计二阶巴特沃斯低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # 应用滤波器
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def moving_average(data, window_size):
    # 构造窗口
    window = np.ones(window_size) / float(window_size)
    # 使用convolve函数对数据进行滤波
    return np.convolve(data, window, 'same')


def apply_filter(df, args):
    if args.filter == 'low_pass':
        cutoff = 0.1  # 截止频率
        fs = 10  # 采样率
        for idx in range(len(df.columns)):
            if df[df.columns[idx]].dtype == 'float64':
                filtered_data = butter_lowpass_filter(df[df.columns[idx]], cutoff, fs, order=2)
                df[df.columns[idx]] = filtered_data
    if args.filter == 'mean':
        for idx in range(len(df.columns)):
            if df[df.columns[idx]].dtype == 'float64':
                filtered_data = moving_average(df[df.columns[idx]], window_size=5)
                df[df.columns[idx]] = filtered_data
    return df
class Dataset_bone_drill(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='A', scale=True, timeenc=0, freq='h', seasonal_patterns=None,args=None):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        drill_folder = '../dataset/bone_drill/'
        dir_list = os.listdir(drill_folder)
        df_raw = None
        for file_name in dir_list:
            df = pd.read_csv(drill_folder + file_name, delimiter=' ')
            df.columns = ['X', 'Y', 'Z', 'A', 'B', 'C']
            # 额外添加一个column在这里，之后再给drop掉等下，这个只是为了求下面的num_train
            df['file_name'] = file_name
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] -= df[col].iloc[0]
            if df_raw is None:
                df_raw = df
            else:
                df_raw = pd.concat([df_raw, df], axis=0)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        df_raw = df_raw[::2]
        # 添加是否使用滤波器
        if self.args.filter != 'no_filter':
            df_raw = apply_filter(df_raw, self.args)

        train_ratio = int(0.6*len(dir_list))
        val_ratio = int(0.8*len(dir_list))
        train_files = dir_list[:train_ratio]
        val_files = dir_list[train_ratio:val_ratio]
        test_files = dir_list[val_ratio:]

        # 只需要确定num_train即可
        num_train = len(df_raw[df_raw['file_name'].isin(train_files)])
        num_test = len(df_raw[df_raw['file_name'].isin(test_files)])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_raw.drop('file_name', axis=1, inplace=True)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values



        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.zeros((self.data_y.shape[0], 1), dtype=float)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_bone_drill_c(Dataset):
    def __init__(self, flag='train', args=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.seed = args.seed

        # init
        # 对于classification任务只会划分为train test
        self.args = args
        assert flag in ['TRAIN', 'TEST']
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        '''
          df_raw.columns: ['date', ...(other features), target feature]
        '''
        self.scaler = StandardScaler()
        drill_folder = '../dataset/'
        file_name = 'df_label.pkl'
        df = pd.read_pickle(drill_folder + file_name)
        df = df.drop(columns='time_stamp')
        df = df[::2] # classifcaion
        a = 1
        # normalizer = Normalizer()
        # df = normalizer.normalize(df)
        self.scaler.fit(df.iloc[:, :self.enc_in].values)
        df.iloc[:, :self.enc_in] = self.scaler.transform(df.iloc[:, :self.enc_in].values)
        # 添加是否使用滤波器
        if self.args.filter != 'no_filter':
            df = apply_filter(df, self.args)

        # 这里也可以直接生成一个index list，只会对index list进行tran test split即可
        val_list = []
        label_list = []
        file_list = []
        for i in range(self.seq_len, df.shape[0], self.seq_len):
            val = df.iloc[i - self.seq_len:i, :self.enc_in].to_numpy().astype('float64')
            label = df.iloc[i]['label']
            file_name = df.iloc[i]['file_name']
            val_list.append(val)
            label_list.append(label)
            file_list.append(file_name)
        df_clean = pd.DataFrame({"value list": val_list, "label": label_list, 'file_name': file_list})
        split_mode = 'file'
        split_mode = 'single'
        if split_mode == 'single':
            x_train, x_test = train_test_split(df_clean, test_size=0.2, random_state=self.seed)
        elif split_mode == 'file':
            name_list = df_clean['file_name'].unique()
            # name_list = name_list[::4]
            train_list, test_list = train_test_split(name_list, test_size=0.2, random_state=self.seed)
            x_train = df_clean[df_clean['file_name'].isin(train_list)]
            # 这里记得改
            x_test = df_clean[df_clean['file_name'].isin(test_list)]
        # 不drop的话还会保存原本的index并形成新的一列
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        ans = x_train['label'].isnull().any()
        ans_t = x_test['label'].isnull().any()
        y_train = x_train['label']
        y_test = x_test['label']
        # 在这里先没有考虑seq_len,对于这个数据集来说最长是128
        if self.flag == 'TRAIN':
            self.data_x = x_train
            self.data_y = y_train
            self.ds_len = len(y_train)
        elif self.flag == 'TEST':
            self.data_x = x_test
            self.data_y = y_test
            self.ds_len = len(y_test)

    def __getitem__(self, index):
        data = self.data_x['value list'].iloc[index].astype('float64')
        data = torch.from_numpy(data)
        # 这里为了照顾这个程序多加了一个【】
        label = torch.tensor([self.data_y[index]]).to(torch.long)
        return data, label

    def __len__(self):
        return self.ds_len

