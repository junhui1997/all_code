from __future__ import division
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
from utils import rotationMatrixToEulerAngles


# For TCN: raw feature
class RawFeatureDataset(Dataset):
    def __init__(self, dataset_name,
                 feature_dir, trail_list,
                 encode_level, sample_rate=1, sample_aug=True,
                 normalization=None):
        super(RawFeatureDataset, self).__init__()

        # 所有video的名称
        # 以video1 来进行举例，在label中可以看到最后的总数是888，采样频率是3Hz，所以是888/3获得下面的296，而kine中是已经采样完的数值是337，需要截断到296
        # mark里面存的是start，和end index对应每个视频文件的，以第一个举例[0,296]
        self.trail_list = trail_list
        # kinematics data sample step 10Hz
        self.video_sampling_step = 3

        from config import kinematics_dir, transcriptions_dir
        self.kinematics_dir = kinematics_dir
        self.transcriptions_dir = transcriptions_dir

        self.sample_rate = sample_rate
        self.sample_aug = sample_aug
        self.encode_level = encode_level

        self.all_feature = []
        self.all_gesture = []
        self.all_kinematics = []
        self.marks = []

        start_index = 0
        for idx in range(len(self.trail_list)):
            
            trail_name = self.trail_list[idx]

            if dataset_name in ['JIGSAWS', 'JIGSAWS_K', 'JIGSAWS_N', 'MISAW', 'Peg_Transfer']:
                data_file = os.path.join(feature_dir, trail_name + '.mat')
            else:
                raise Exception('Invalid Dataset Name!') 
            
            trail_data = scipy.io.loadmat(data_file)
            # vision features
            trail_feature = trail_data['A']

            trail_gesture = trail_data['Y']
            trail_len = trail_gesture.shape[0]

            # kinematics features
            # 以图像数据截断了kinetic data，原本的kinetic data稍微长一点
            trail_kinematics = self._load_kinematics(trail_name, trail_len)

            assert trail_len == trail_kinematics.shape[0]

            self.all_feature.append(trail_feature)
            self.all_gesture.append(trail_gesture)
            self.all_kinematics.append(trail_kinematics)

            self.marks.append([start_index, start_index + trail_len])
            start_index += trail_len

        self.all_feature = np.concatenate(self.all_feature)
        self.all_gesture = np.concatenate(self.all_gesture)
        self.all_kinematics = np.concatenate(self.all_kinematics)

        # Normalization
        if normalization is not None:
            self.feature_means = normalization[0][0]
            self.feature_stds = normalization[1][0]
            self.kinematics_means = normalization[0][1]
            self.kinematics_stds = normalization[1][1]

            self.all_feature = self.all_feature - self.feature_means
            self.all_feature = self.all_feature / self.feature_stds
            self.all_kinematics = self.all_kinematics - self.kinematics_means
            self.all_kinematics = self.all_kinematics / self.kinematics_stds

        else:
            self.feature_means = self.all_feature.mean(0)
            self.feature_stds = self.all_feature.std(0)
            self.kinematics_means = self.all_kinematics.mean(0)
            self.kinematics_stds = self.all_kinematics.std(0)

            self.all_feature = self.all_feature - self.feature_means
            self.all_feature = self.all_feature / self.feature_stds
            self.all_kinematics = self.all_kinematics - self.kinematics_means
            self.all_kinematics = self.all_kinematics / self.kinematics_stds


    def __len__(self):
        if self.sample_aug:
            return len(self.trail_list) * self.sample_rate
        else:
            # 注意这里len的返回值
            return len(self.trail_list)

    def __getitem__(self, idx):
        # 这里默认是false
        if self.sample_aug:
            trail_idx = idx // self.sample_rate
            sub_idx = idx % self.sample_rate
        else:
            trail_idx = idx
            sub_idx = 0 

        trail_name = self.trail_list[trail_idx]

        # 获取了相应video的起止
        start = self.marks[trail_idx][0]
        end = self.marks[trail_idx][1]

        # 这里去获取相应的feature和gesture
        feature = self.all_feature[start:end,:]
        gesture = self.all_gesture[start:end,:]
        kinematics = self.all_kinematics[start:end,:]
        # sample_rate默认也是1,下面list的含义是从0~列表结束，重新采样，采样频率为sample rate
        # 如果sample = 1，相当于没有做出任何改变
        feature = feature[sub_idx::self.sample_rate]
        gesture = gesture[sub_idx::self.sample_rate]
        kinematics = kinematics[sub_idx::self.sample_rate]

        trail_len = gesture.shape[0]
        # 这个是为了填充到它所需要的长度上面去
        padded_len = int(np.ceil(trail_len / 
                          (2**self.encode_level)))*2**self.encode_level
        # mask：shape (padded_len,1)
        mask = np.zeros([padded_len, 1])
        mask[0:trail_len] = 1

        padded_feature = np.zeros([padded_len, feature.shape[1]])
        padded_feature[0:trail_len] = feature

        padded_kinematics = np.zeros([padded_len, kinematics.shape[1]])
        padded_kinematics[0:trail_len] = kinematics

        padded_gesture = np.zeros([padded_len, 1])-1
        padded_gesture[0:trail_len] = gesture

        # padded_feature这样相当于是将形状扩充了一下，扩充的部分为0，其余部分和原有的feature一致
        return {'feature': padded_feature,
                'gesture': padded_gesture,
                'kinematics': padded_kinematics,
                'mask': mask,
                'name': trail_name,
                'trail_len':trail_len}

    def get_means(self):
        return [self.feature_means, self.kinematics_means]

    def get_stds(self):
        return [self.feature_stds, self.kinematics_stds]

    def _load_kinematics(self, video_id, _count):
        print("Preloading kinematics from video {}...".format(video_id))
        kinematics = []
        kinematics_dir = os.path.join(self.kinematics_dir, video_id + ".txt")
        ###
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # print current file directory
        print(current_dir)
        ###
        kinematics_temp = np.loadtxt(kinematics_dir, dtype=np.float32)
        transcriptions_dir = os.path.join(self.transcriptions_dir, video_id + ".txt")
        # offset = int(np.loadtxt(transcriptions_dir, dtype=np.str)[0][0])
        # for idx in range(_count):
        #     if (idx*self.video_sampling_step + offset) < len(kinematics_temp):
        #         kinematics.append(kinematics_temp[idx*self.video_sampling_step + offset])
        #     else:
        #         break
        for idx in range(_count):
            kinematics.append(kinematics_temp[idx])
            
        kin_data = np.array(kinematics)
        # left_euler = np.array(
        #     [rotationMatrixToEulerAngles(kin_data[i][41:50].reshape(3, 3)) for i in range(len(kin_data))]
        # )
        # right_euler = np.array(
        #     [rotationMatrixToEulerAngles(kin_data[i][60:69].reshape(3, 3)) for i in range(len(kin_data))]
        # )

        # new_kin_data = np.hstack((kin_data[:, 38:41], left_euler, kin_data[:, 50:57],
        #                           kin_data[:, 57:60], right_euler, kin_data[:, 69:76]))

        return kin_data

