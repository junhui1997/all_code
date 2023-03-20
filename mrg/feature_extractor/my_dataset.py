from __future__ import division
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
from transform_list import transform_train, transform_test
import scipy.io
import os

import torch
from PIL import Image
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')



# For TCN: raw feature
class RawFeatureDataset(Dataset):
    def __init__(self,  args, trail_list=None, enc=None, flag='train'):
        super(RawFeatureDataset, self).__init__()

        self.flag = flag
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

        trail_list_txt = [file_name + '.txt' for file_name in trail_list]
        sampled_df = None
        for file_name in trail_list_txt:
            df_n = df.loc[(df['file_name'] == file_name)]
            df_n = df_n.loc[(df['gesture'] != 'None')]
            # to ensure the seq len does not exceed limit
            sample_rate = len(df_n) // args.seq_limit + 1
            # sample_rate = 2
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

        self.df = sampled_df
        self.args = args
        self.gestures = self.enc.transform(self.df['gesture'])




    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        trail_txt = self.df.iloc[idx]['file_name']
        trail = trail_txt.split('.')[0]
        frame_num = int(self.df.iloc[idx]['frame'])
        file_name = "{}_capture1_frame_{}".format(trail,frame_num)
        img = np.array(Image.open('{}/{}.jpg'.format(self.image_floder, file_name)))
        img = img / 255
        if self.flag =='train':
            img = transform_train(img)
        else:
            img = transform_test(img)
        img.permute(2, 0, 1)
        img = img.to(torch.float)



        gesture = self.gestures[idx]
        gesture = torch.tensor(gesture).to(torch.long)



        # padded_feature这样相当于是将形状扩充了一下，扩充的部分为0，其余部分和原有的feature一致
        return img, gesture, trail
        # return {'feature': img,
        #         'gesture': gesture,
        #         'trail': trail,
        #         }

    def get_enc(self):
        return self.enc

