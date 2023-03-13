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
    def __init__(self,  args):
        super(RawFeatureDataset, self).__init__()


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

        # 获取所有file的名字，并去除txt
        self.trail_list = list(df['file_name'].unique())
        self.trail_list = [trail.split('.')[0] for trail in self.trail_list]
        self.df = df
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
            img = transform_test(img)
            img.permute(2, 0, 1)
            img = img.to(torch.float)
            img = img.unsqueeze(0)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)


        # padded_feature这样相当于是将形状扩充了一下，扩充的部分为0，其余部分和原有的feature一致
        return {'feature': imgs,
                'trail': trail,
                }


