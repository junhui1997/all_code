from __future__ import division
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os

import torch
from PIL import Image
from module_box.transform_list import transform_train,transform_test
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
from utils import rotationMatrixToEulerAngles


# For TCN: raw feature
class RawFeatureDataset(Dataset):
    def __init__(self, dataset_name, trail_list, args,
                 normalization=None , enc = None):
        super(RawFeatureDataset, self).__init__()

        # 所有video的名称
        self.trail_list = trail_list
        # 对于每个file都单独计算不同的sample
        # 先读取文件，直接所有的东西都放进init里面，不再分prepare data了
        folder = '../../../jigsaw/'
        if args.task_name == 'kt':
            df = pd.read_pickle(folder + 'Knot_Tying.pkl')
            self.image_floder = '../../../jigsaw/video_slice/Knot_Tying/'
        elif args.task_name == 'np':
            df = pd.read_pickle(folder + 'Needle_Passing.pkl')
            self.image_floder = '../../../jigsaw/video_slice/Needle_Passing/'
        elif args.task_name == 'su':
            df = pd.read_pickle(folder + 'Suturing.pkl')
            self.image_floder = '../../../jigsaw/video_slice/Suturing/'

        trail_list_txt = [file_name+'.txt' for file_name in trail_list]
        sampled_df = None
        for file_name in trail_list_txt:
            df_n = df.loc[(df['file_name'] == file_name)]
            df_n = df_n.loc[(df['gesture'] != 'None')]
            # to ensure the seq len does not exceed limit
            sample_rate = len(df_n)//args.seq_limit + 1
            df_n = df_n[::sample_rate]
            if sampled_df is None:
                sampled_df = df_n
            else:
                sampled_df = pd.concat([sampled_df, df_n], axis=0)
        sampled_df = sampled_df.reset_index(drop=True)
        if enc is None:
            self.enc = LabelEncoder()
            self.enc.fit_transform(sampled_df['gesture'])
        else:
            self.enc = enc
        self.class_name = self.enc.inverse_transform([i for i in range(len(sampled_df['gesture'].unique()))])


        # Normalization
        if normalization is not None:
            self.kinematics_means = normalization[0][1]
            self.kinematics_stds = normalization[1][1]
            for i in range(args.enc_in):
                sampled_df[3 + i] = sampled_df[3 + i].astype(float)
                mean = self.kinematics_means[i]
                std = self.kinematics_stds[i]
                sampled_df[3 + i] = sampled_df[3 + i].apply(lambda x: (x - mean) / std)

        else:
            all_mean = []
            all_std = []
            for i in range(args.enc_in):
                sampled_df[3 + i] = sampled_df[3 + i].astype(float)
                mean = sampled_df[3 + i].mean()
                std = sampled_df[3 + i].std()
                sampled_df[3 + i] = sampled_df[3 + i].apply(lambda x: (x - mean) / std)
                all_mean.append(mean)
                all_std.append(std)
            self.kinematics_means = all_mean
            self.kinematics_stds = all_std
        self.df = sampled_df
        self.args = args




    def __len__(self):
        return len(self.trail_list)

    def __getitem__(self, idx):

        trail = self.trail_list[idx]
        df_item = self.df.loc[(self.df['file_name'] == (trail+'.txt'))]
        trail_len = len(df_item)
        imgs = None
        for i in range(trail_len):
            file_name = "{}_capture1_frame_{}".format(trail,
                                                      int(df_item.iloc[i]['frame']))
            img = np.array(Image.open('{}/{}.jpg'.format(self.image_floder, file_name)))
            img = img / 255
            img = transform_train(img)
            img.permute(2, 0, 1)
            img = img.to(torch.float)
            img = img.unsqueeze(0)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)
        #imgs = torch.rand(1,1,1)
        kinematics = df_item.iloc[:, 11:11 + self.args.enc_in].to_numpy().astype('float64')
        gesture = self.enc.transform(df_item['gesture'])
        gesture = torch.tensor(gesture).to(torch.long)
        # padded_feature这样相当于是将形状扩充了一下，扩充的部分为0，其余部分和原有的feature一致
        return {'feature': imgs,
                'gesture': gesture,
                'kinematics': kinematics,
                'trail_len':trail_len}

    def get_means(self):
        return [0, self.kinematics_means]

    def get_stds(self):
        return [0, self.kinematics_stds]

    def get_enc(self):
        return self.enc

    def get_class_name(self):
        return self.class_name

