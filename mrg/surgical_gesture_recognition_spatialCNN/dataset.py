# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

import torch
import torch.utils.data as data
import torchvision
from transforms import Stack, ToTorchFormatTensor

from PIL import Image
import os
import numpy as np
import scipy.io as scio
from numpy.random import randint

from util import rotationMatrixToEulerAngles
from util import kin_mean, kin_std


class GestureRecord(object):
    def __init__(self, g_segments, snippet_length=16, min_overlap=1):
        self.start_frames = []
        self.accumulated_snippet_counts = [0]
        self.num_unique_snippets = 0
        _accumulated_snippet_count = 0
        for s in g_segments:
            if s[1] - s[0] + 1 >= min_overlap:  # at least one complete snippet in segment
                start = s[0] - snippet_length + min_overlap
                end = s[1] - snippet_length + 1

                self.start_frames.append(start)
                _accumulated_snippet_count += end - start + 1
                self.accumulated_snippet_counts.append(_accumulated_snippet_count)
                self.num_unique_snippets += (s[1] - start) // snippet_length

    def sample_idx(self):
        idx = randint(self.snippet_count())

        # transform to video-level frame no.
        i = 0
        while not (idx < self.accumulated_snippet_counts[i + 1]):
            i += 1
        idx = idx - self.accumulated_snippet_counts[i] + self.start_frames[i]
        return idx

    def snippet_count(self):
        return self.accumulated_snippet_counts[-1]


class GestureDataSet(data.Dataset):
    def __init__(self, root_path, list_of_list_files, transcriptions_dir, gesture_ids, split_set,
                 kinematics_dir=None, vision_dir=None, snippet_length=16, min_overlap=1, video_sampling_step=6,
                 modality='RGB', image_tmpl='{:d}.png', video_suffix="_capture2",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None, load_to_RAM=True, transpose_img=True):

        self.root_path = root_path
        self.list_of_list_files = list_of_list_files
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.split_set = split_set
        self.kinematics_dir = kinematics_dir
        self.vision_dir = vision_dir
        self.snippet_length = snippet_length
        self.min_overlap = min_overlap
        self.video_sampling_step = video_sampling_step
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels
        self.transform = transform
        self.normalize = normalize
        self.load_to_RAM = load_to_RAM
        self.transpose_img = transpose_img

        self.gesture_dict = {}  # for each gesture, save which segments of which video belong to that gesture
        self.min_g_count = 0  # for each gesture, there are at least <min_g_count> non-overlapping snippets in the dataset

        self.gesture_sequence_per_video = {}
        self.image_data = {}

        # TODO
        self.kinematics_data = {}
        # self.vision_data = {}

        self._parse_list_files(list_of_list_files)

    def _parse_list_files(self, list_of_list_files):
        for list_file in list_of_list_files:
            videos = [(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)]
            for video in videos:
                video_id = video[0]
                frame_count = int(video[1])

                gestures_file = os.path.join(self.transcriptions_dir, video_id + ".txt")
                gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                            for x in open(gestures_file)]
                           # [start_frame, end_frame, gesture_id]
                """
                for i in range(len(gestures)):
                    if i + 1 < len(gestures):
                        assert (gestures[i][1] == gestures[i + 1][0] - 1)
                    else:
                        assert (gestures[i][1] < frame_count)
                """

                # adjust indices to temporal downsampling (specified by "video_sampling_step")
                _frame_count = frame_count // self.video_sampling_step
                _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                               '{:d}.png'.format(_frame_count))
                if not os.path.isfile(_last_rgb_frame):
                    _frame_count = _frame_count - 1

                for i in range(len(gestures)):
                    gestures[i][0] = int(round(gestures[i][0] / self.video_sampling_step))
                for i in range(len(gestures)):
                    if i + 1 < len(gestures):
                        gestures[i][1] = gestures[i + 1][0] - 1
                    else:
                        next_frame = gestures[i][1] + 1
                        next_frame = int(round(next_frame / self.video_sampling_step))
                        gestures[i][1] = next_frame - 1

                        if gestures[i][1] == _frame_count:
                            gestures[i][1] -= 1
                        assert (gestures[i][1] < _frame_count)

                if self.return_dense_labels:
                    # save sequence of gesture labels for this video
                    labels = []
                    start_frames = []
                    for i in range(len(gestures)):
                        labels.append(gestures[i][2])
                        start_frames.append(gestures[i][0])
                    last_frame = gestures[-1][1]
                    start_frames.append(last_frame + 1)

                    assert(video_id not in self.gesture_sequence_per_video)
                    self.gesture_sequence_per_video[video_id] = (labels, start_frames)

                # update gesture_dict data structure
                while len(gestures) > 0:
                    g_id = gestures[0][2]
                    g_segments = [(gestures[0][0], gestures[0][1])]
                    del gestures[0]

                    # find segments referring to the same gesture
                    _gestures = []
                    for i in range(len(gestures)):
                        if gestures[i][2] == g_id:
                            g_segments.append((gestures[i][0], gestures[i][1]))
                        else:
                            _gestures.append(gestures[i])
                    del gestures
                    gestures = _gestures

                    record = GestureRecord(g_segments, self.snippet_length, self.min_overlap)
                    if record.num_unique_snippets > 0:
                        if g_id not in self.gesture_dict:
                            self.gesture_dict[g_id] = {}
                            self.gesture_dict[g_id]['count'] = 0
                            self.gesture_dict[g_id]['segments_per_video'] = {}
                        assert (video_id not in self.gesture_dict[g_id]['segments_per_video'])
                        self.gesture_dict[g_id]['segments_per_video'][video_id] = record
                        self.gesture_dict[g_id]['count'] += record.num_unique_snippets

                if self.load_to_RAM:
                    self._preload_images(video_id, _frame_count)

                self._preload_kinematics(video_id,_frame_count)
                # self._preload_vision(video_id)

        for g_id in self.gesture_dict:
            assert(g_id in self.gesture_ids)
        if len(self.gesture_dict.keys()) != len(self.gesture_ids):
            missing_gestures = [k for k in self.gesture_ids if k not in self.gesture_dict]
            print("Warning! Gestures missing in dataset:", missing_gestures)
            self.gesture_ids = sorted(list(self.gesture_dict.keys()))

        for g_id in self.gesture_dict:
            video_ids = sorted(self.gesture_dict[g_id]['segments_per_video'])
            self.gesture_dict[g_id]['videos'] = video_ids

        self.min_g_count = np.min(np.array([self.gesture_dict[k]['count'] for k in self.gesture_dict]))

    def _preload_images(self, video_id, _frame_count):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in range(_frame_count):
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data[video_id] = images

    def _preload_kinematics(self, video_id, _frame_count):
        print("Preloading kinematics from video {}...".format(video_id))
        kinematics = []
        kinematics_dir = os.path.join(self.kinematics_dir, video_id + ".txt")
        kinematics_temp = np.loadtxt(kinematics_dir, dtype=np.float32)
        for idx in range(_frame_count):
            # TODO
            if idx*self.video_sampling_step < len(kinematics_temp):
                kinematics.append(kinematics_temp[idx*self.video_sampling_step])
        self.kinematics_data[video_id] = kinematics

    # def _preload_vision(self, video_id):
    #     print("Preloading vision features from video {}...".format(video_id))
    #     vision_dir = os.path.join(self.vision_dir, 'Split_'+str(self.split_set+1), video_id + ".avi.mat")
    #     vision_data = scio.loadmat(vision_dir)
    #     self.vision_data[video_id] = vision_data['A']

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            img = Image.open(os.path.join(directory, self.image_tmpl.format(idx + 1))).convert('RGB')
            return [img]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx + 1))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx + 1))).convert('L')
            return [x_img, y_img]

    def __getitem__(self, index):
        g_id, video_id, idx = self._sample_gesture_snippet(index)
        img_data, kin_data, indices = self.get(video_id, idx)
        if not self.return_dense_labels:
            target = self._to_gesture_label(g_id)
        else:
            target = self._get_snippet_labels(video_id, indices)
            assert(target[-1] == self._to_gesture_label(g_id))

        # print("debug")
        # print(index)
        # print(g_id, video_id, idx, indices)
        # print(target)
        # print(img_data.shape)
        # print(kin_data)
        # print("debug")
        # exit()

        # 19
        # G2 Suturing_C002 275 [275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290]
        # [5 5 5 5 5 5 1 1 1 1 1 1 1 1 1 1]
        # torch.Size([3, 16, 224, 224])

        return img_data, kin_data, target

    def _sample_gesture_snippet(self, index):
        g_id = self.gesture_ids[index // self.min_g_count]

        video_id = self.gesture_dict[g_id]['videos'][randint(len(self.gesture_dict[g_id]['videos']))]
        snippet_idx = self.gesture_dict[g_id]['segments_per_video'][video_id].sample_idx()

        return g_id, video_id, snippet_idx

    def get(self, video_id, idx):
        images, indices = self.get_snippet(video_id, idx)

        if self.return_3D_tensor:
            images = self.transform(images)
            images = [torchvision.transforms.ToTensor()(img) for img in images]
            if self.modality == 'RGB':
                images = torch.stack(images, 0)
            elif self.modality == 'Flow':
                _images = []
                for i in range(len(images) // 2):
                    _images.append(torch.cat([images[i], images[i + 1]], 0))
                images = torch.stack(_images, 0)
            images = self.normalize(images)
            images = images.view(((self.snippet_length,) + images.size()[-3:]))
            if self.transpose_img:
                images = images.permute(1, 0, 2, 3)
            data = images
        else:
            transform = torchvision.transforms.Compose([
                self.transform,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                self.normalize,
            ])
            data = transform(images)

        # TODO kinematics data
        kin_data = []
        for _index in indices:
            kin_data.append(self.kinematics_data[video_id][_index])

        kin_data = np.array(kin_data)
        left_euler = np.array(
            [rotationMatrixToEulerAngles(kin_data[i][41:50].reshape(3, 3)) for i in range(len(kin_data))]
        )
        right_euler = np.array(
            [rotationMatrixToEulerAngles(kin_data[i][60:69].reshape(3, 3)) for i in range(len(kin_data))]
        )

        new_kin_data = np.hstack((kin_data[:, 38:41], left_euler, kin_data[:, 50:57],
                                  kin_data[:, 57:60], right_euler, kin_data[:, 69:76]))

        for i in range(len(new_kin_data)):
            new_kin_data[i] = (new_kin_data[i] - kin_mean)/kin_std

        new_kin_data = torch.Tensor(new_kin_data)

        # TODO vision data

        return data, new_kin_data, indices

    def get_snippet(self, video_id, idx):
        snippet = list()
        indices = list()
        for _ in range(self.snippet_length):
            _idx = max(idx, 0)  # padding if required
            if self.load_to_RAM:
                if self.modality == 'RGB':
                    imgs = self.image_data[video_id][_idx: _idx + 1]
                elif self.modality == 'Flow':
                    i = _idx * 2
                    imgs = self.image_data[video_id][i: i + 2]
            else:
                img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
                imgs = self._load_image(img_dir, _idx)
            snippet.extend(imgs)
            indices.append(_idx)
            idx += 1
        return snippet, indices

    def _get_snippet_labels(self, video_id, indices):
        assert self.return_dense_labels
        labels, start_frames = self.gesture_sequence_per_video[video_id]
        target = []
        for idx in indices:
            i = 0
            while idx >= start_frames[i + 1]:
                i += 1
            target.append(self._to_gesture_label(labels[i]))
        return np.array(target, dtype=np.int32)

    def _to_gesture_label(self, gesture_id):
        return self.gesture_ids.index(gesture_id)

    def __len__(self):
        return self.min_g_count * len(self.gesture_ids)


class SequentialGestureDataSet(data.Dataset):
    def __init__(self, root_path, transcriptions_dir, gesture_ids, video_id, frame_count,
                 kinematics_dir=None, snippet_length=16, video_sampling_step=6, snippet_sampling_step=6,
                 modality='RGB', image_tmpl='{:d}.png', video_suffix="_capture2",
                 return_3D_tensor=True, transform=None, normalize=None, load_to_RAM=True,
                 transpose_img=True):
                 # sample snippets every *video_sampling_step*th frame,
                 # but within each snippet, video is sampled at every *snippet_sampling_step*th frame

        self.root_path = root_path
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.video_id = video_id
        self.frame_count = frame_count
        self.kinematics_dir = kinematics_dir
        self.snippet_length = snippet_length
        self.video_sampling_step = video_sampling_step
        self.snippet_sample_step = snippet_sampling_step
        self.sample_from_full_resolution = snippet_sampling_step != video_sampling_step
        if self.sample_from_full_resolution:
            assert load_to_RAM is False  # implementation cannot handle this at the moment
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.transform = transform
        self.normalize = normalize
        self.load_to_RAM = load_to_RAM
        self.transpose_img = transpose_img

        self.labels = []
        self.start_frames = []
        self.offset = 0
        self.length = 0

        self.image_data = None
        self.kinematics_data = None

        self._read_transcription()

    def _read_transcription(self):
        gestures_file = os.path.join(self.transcriptions_dir, self.video_id + ".txt")
        gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                    for x in open(gestures_file)]

        _frame_count = self.frame_count // self.video_sampling_step
        if not self.sample_from_full_resolution:
            _last_rgb_frame = os.path.join(self.root_path, self.video_id + self.video_suffix,
                                           '{:d}.png'.format(_frame_count))
            if not os.path.isfile(_last_rgb_frame):
                _frame_count = _frame_count - 1
        else:
            _last_rgb_frame = os.path.join(self.root_path, self.video_id + self.video_suffix,
                                           '{:d}.png'.format(self.frame_count - 1))
            assert(os.path.isfile(_last_rgb_frame)), \
                "Sampling snippets with step {} requires sampling from the temporally fully resolved data (30 fps)."\
                    .format(self.video_sampling_step)

        for i in range(len(gestures)):
            self.labels.append(self.gesture_ids.index(gestures[i][2]))
            self.start_frames.append(int(round(gestures[i][0] /
                                               (1 if self.sample_from_full_resolution else self.video_sampling_step))))

            if i == len(gestures) - 1:
                # calculate index of last annotated frame
                last_frame = int(round((gestures[i][1] + 1) /
                                       (1 if self.sample_from_full_resolution else self.video_sampling_step))) - 1
                if not self.sample_from_full_resolution:
                    if last_frame == _frame_count:
                        last_frame -= 1
                    assert (last_frame < _frame_count)

                # self.length = math.ceil((last_frame - self.start_frames[0] + 1) /
                #                         (self.video_sampling_step if self.sample_from_full_resolution else 1))
                # TODO
                self.length = (last_frame - self.start_frames[0] + 1) //\
                              (self.video_sampling_step if self.sample_from_full_resolution else 1)

                self.start_frames.append(last_frame + 1)

        self.offset = self.start_frames[0]

        if self.load_to_RAM:
            self._preload_images(_frame_count)
        self._preload_kinematics(_frame_count)

    def _preload_images(self, _frame_count):
        print("Preloading images from video {}...".format(self.video_id))
        images = []
        img_dir = os.path.join(self.root_path, self.video_id + self.video_suffix)
        for idx in range(_frame_count):
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data = images

    def _preload_kinematics(self, _frame_count):
        print("Preloading kinematics from video {}...".format(self.video_id))
        kinematics = []
        kinematics_dir = os.path.join(self.kinematics_dir, self.video_id + ".txt")
        kinematics_temp = np.loadtxt(kinematics_dir, dtype=np.float32)
        if self.sample_from_full_resolution:
            self.kinematics_data = kinematics_temp
        else:
            for idx in range(_frame_count):
                # TODO
                if idx*self.video_sampling_step < len(kinematics_temp):
                    kinematics.append(kinematics_temp[idx*self.video_sampling_step])
            self.kinematics_data = kinematics

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            img = Image.open(os.path.join(directory, self.image_tmpl.format(idx + 1))).convert('RGB')
            return [img]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx + 1))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx + 1))).convert('L')
            return [x_img, y_img]

    def __getitem__(self, index):
        idx = (index * (self.video_sampling_step if self.sample_from_full_resolution else 1)) + self.offset
        img_snippet, kin_data = self.get(idx - (self.snippet_length - 1) *
                                     (self.snippet_sample_step if self.sample_from_full_resolution else 1))

        i = 0
        while idx >= self.start_frames[i + 1]:
            i += 1
        target = self.labels[i]

        return img_snippet, kin_data, target

    def get(self, idx):
        images, indices = self.get_snippet(idx)

        if self.return_3D_tensor:
            images = self.transform(images)
            images = [torchvision.transforms.ToTensor()(img) for img in images]
            if self.modality == 'RGB':
                images = torch.stack(images, 0)
            elif self.modality == 'Flow':
                _images = []
                for i in range(len(images) // 2):
                    _images.append(torch.cat([images[i], images[i + 1]], 0))
                images = torch.stack(_images, 0)
            images = self.normalize(images)
            images = images.view(((self.snippet_length,) + images.size()[-3:]))
            if self.transpose_img:
                images = images.permute(1, 0, 2, 3)
            data = images
        else:
            if self.normalize is not None:
                transform = torchvision.transforms.Compose([
                    self.transform,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    self.normalize,
                ])
            else:
                if self.transform is not None:
                    transform = torchvision.transforms.Compose([
                        self.transform,
                        Stack(roll=False),
                        ToTorchFormatTensor(div=True)
                    ])
                else:
                    transform = torchvision.transforms.Compose([
                        Stack(roll=False),
                        ToTorchFormatTensor(div=True)
                    ])

            data = transform(images)
        # TODO kinematics data
        kin_data = []
        for _index in indices:
            kin_data.append(self.kinematics_data[_index])
        
        kin_data = np.array(kin_data)
        left_euler = np.array(
            [rotationMatrixToEulerAngles(kin_data[i][41:50].reshape(3, 3)) for i in range(len(kin_data))]
        )
        right_euler = np.array(
            [rotationMatrixToEulerAngles(kin_data[i][60:69].reshape(3, 3)) for i in range(len(kin_data))]
        )

        new_kin_data = np.hstack((kin_data[:, 38:41], left_euler, kin_data[:, 50:57],
                                  kin_data[:, 57:60], right_euler, kin_data[:, 69:76]))

        for i in range(len(new_kin_data)):
            new_kin_data[i] = (new_kin_data[i] - kin_mean)/kin_std

        new_kin_data = torch.Tensor(new_kin_data)

        return data, new_kin_data

    def get_snippet(self, idx):
        snippet = list()
        indices = list()
        for _ in range(self.snippet_length):
            _idx = max(idx, 0)  # padding if required
            if self.load_to_RAM:
                if self.modality == 'RGB':
                    imgs = self.image_data[_idx: _idx + 1]
                elif self.modality == 'Flow':
                    i = _idx * 2
                    imgs = self.image_data[i: i + 2]
            else:
                img_dir = os.path.join(self.root_path, self.video_id + self.video_suffix)
                imgs = self._load_image(img_dir, _idx)
            snippet.extend(imgs)
            indices.append(_idx)
            idx += (self.snippet_sample_step if self.sample_from_full_resolution else 1)
        return snippet, indices

    def __len__(self):
        return self.length
