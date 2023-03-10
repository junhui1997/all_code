# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

import os.path
import string
import math
import numpy as np
import torch
import torchvision
import scipy.ndimage as ndimage

from test_opts import parser
from dataset import SequentialGestureDataSet
from models import GestureClassifier
from metrics import accuracy, average_F1, edit_score, overlap_f1
from transforms import GroupNormalize, GroupScale, GroupCenterCrop
from util import gestures_SU, gestures_NP, gestures_KT, splits_LOSO, splits_LOUO, splits_LOUO_NP
from torch.utils.tensorboard import SummaryWriter
from util import select_n_random

def eval_exp(args):

    base_dir = os.path.join(args.model_dir, args.exp, args.eval_scheme)
    print("Evaluate " + base_dir)
    tb_writer = SummaryWriter(os.path.join(base_dir, "tensorboard"))

    if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
        args.data_path = args.data_path.format(args.task)
    if len([t for t in string.Formatter().parse(args.video_lists_dir)]) > 1:
        args.video_lists_dir = args.video_lists_dir.format(args.task)
    if len([t for t in string.Formatter().parse(args.transcriptions_dir)]) > 1:
        args.transcriptions_dir = args.transcriptions_dir.format(args.task)

    gesture_ids = None
    if args.task == "Suturing":
        gesture_ids = gestures_SU
    elif args.task == "Needle_Passing":
        gesture_ids = gestures_NP
    elif args.task == "Knot_Tying":
        gesture_ids = gestures_KT
    num_class = len(gesture_ids)

    splits = None
    if args.eval_scheme == 'LOSO':
        splits = splits_LOSO
    elif args.eval_scheme == 'LOUO':
        if args.task == "Needle_Passing":
            splits = splits_LOUO_NP
        else:
            splits = splits_LOUO

    video_ids = list()  # find all videos
    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    video_lists = list(map(lambda x: os.path.join(lists_dir, x), splits))
    for list_file in video_lists:
        video_ids.extend([x.strip().split(',')[0] for x in open(list_file)])

    epochs = args.model_no
    if epochs is None:
        # find epochs to evaluate
        epochs = []
        d = [f for f in os.listdir(os.path.join(base_dir, str(0))) if not f.startswith('.')]
        if len(d) != 1:
            print(d)
            print("Unclear which epochs to evaluate.")
            return -1
        for file in os.listdir(os.path.join(base_dir, str(0), d[0])):
            if file.startswith("model_"):
                epochs.append(int(file.split('_')[-1].split('.')[0]))
        epochs = sorted(epochs)

    eval_type = "plain"
    if args.sliding_window:
        eval_type = "window"
        if args.look_ahead is not None:
            eval_type += "_" + str(args.look_ahead)
    eval_freq = 30 // args.video_sampling_step

    for epoch in epochs:
        print("Evaluate model no. " + str(epoch))

        logits_file = os.path.join(args.model_dir, "Eval", args.eval_scheme, "{}Hz".format(eval_freq),
                                   "logits", args.exp, "{}.pth.tar".format(epoch))
        labels_file = os.path.join(args.model_dir, "Eval", args.eval_scheme, "{}Hz".format(eval_freq), "labels.pth.tar")
        if not os.path.exists(logits_file):
            logits_dir = os.path.join(args.model_dir, "Eval", args.eval_scheme, "{}Hz".format(eval_freq),
                                      "logits", args.exp)
            if not os.path.exists(logits_dir):
                os.makedirs(logits_dir)

            _labels_file = None if os.path.exists(labels_file) else labels_file
            success = get_logits(args, splits, gesture_ids, epoch, logits_file, _labels_file,tb_writer=tb_writer)
            if success == -1:
                continue

        results_dir = os.path.join(args.model_dir, "Eval", args.eval_scheme, "{}Hz".format(eval_freq), eval_type,
                                   args.exp)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, "{}.pth.tar".format(epoch))

        results = {}
        for video_id in video_ids:
            success = calculate_metrics(args, video_id, num_class, logits_file, labels_file, results)
            if success == -1:
                return

        # print(results)
        # exit()

        # accumulate results over all videos
        overall_results = {}
        for metric in ['acc', 'avg_f1', 'edit', 'f1_10', 'f1_25', 'f1_50']:
            values = []
            for video_id in results:
                values.append(results[video_id][metric])
            overall_results[metric] = np.mean(values), np.std(values)

        conf_mat = np.zeros([num_class, num_class], dtype=np.int64)
        for video_id in results:
            conf_mat += results[video_id]['conf_mat']
        overall_results['conf_mat'] = conf_mat
        results['overall'] = overall_results

        acc = overall_results['acc']
        avg_f1 = overall_results['avg_f1']
        edit = overall_results['edit']
        f1_10 = overall_results['f1_10']
        msg = "{}\t{:.4f}+/-{:.4f}\t{:.4f}+/-{:.4f}\t{:.4f}+/-{:.4f}\t{:.4f}+/-{:.4f}"\
            .format(epoch, acc[0], acc[1], avg_f1[0], avg_f1[1], edit[0], edit[1], f1_10[0], f1_10[1])
        print(msg)

        torch.save(results, results_file)


def calculate_metrics(args, video_id, num_class, logits_file, labels_file, results):

    if args.video_sampling_step == 6:  # 5Hz
        snippet_length = 16
    elif args.video_sampling_step == 15:  # 2Hz
        snippet_length = 7
    elif args.video_sampling_step == 3:  # 10Hz
        snippet_length = 32
    else:
        print("Evaluation at temporal resolutions other than 2Hz, 5Hz, or 10Hz is not implemented.")
        return -1

    labels = torch.load(labels_file)[video_id]
    logits = torch.load(logits_file)[video_id]
    assert(labels.shape[0] == logits.shape[0])
    frame_wise_predictions = len(logits.shape) == 2  # predictions from 2D CNN

    # calculate final predictions

    P = np.array([], dtype=np.int64)
    _P = torch.zeros([labels.shape[0], num_class], dtype=torch.float)
    Y = labels
    # for i in range(logits.shape[0]):
    #     if frame_wise_predictions:
    #         output = logits[i, :]
    #         output = output.view(-1, output.size(0))
    #     else:
    #         output = logits[i, :, :]
    #         output = output.view(-1, output.size(0), output.size(1))
    #         if not args.sliding_window:
    #             output = output[:, :, -1]  # consider only final prediction
    predicted = torch.nn.Softmax(dim=1)(logits)
    
    _, predicted = torch.max(predicted, 1)
    P = np.append(P, predicted.cpu().numpy())

    eval_record = {}
    eval_record['P'] = P
    eval_record['Y'] = Y

    acc = accuracy(P, Y)
    avg_f1, conf_mat = average_F1(P, Y, n_classes=num_class)
    edit = edit_score(P, Y)
    f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
    f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
    f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)

    msg = "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(video_id, acc, avg_f1, edit, f1_10)
    print(msg)

    eval_record['acc'] = acc
    eval_record['avg_f1'] = avg_f1
    eval_record['conf_mat'] = conf_mat
    eval_record['edit'] = edit
    eval_record['f1_10'] = f1_10
    eval_record['f1_25'] = f1_25
    eval_record['f1_50'] = f1_50
    results[video_id] = eval_record


def get_logits(args, splits, gesture_ids, model_no, logits_file, labels_file=None, tb_writer=None):

    base_dir = os.path.join(args.model_dir, args.exp, args.eval_scheme)

    logits_per_video = {}
    labels_per_video = None
    if labels_file is not None:
        labels_per_video = {}
    for split in range(len(splits)):
        # find model
        d = [f for f in os.listdir(os.path.join(base_dir, str(split))) if not f.startswith('.')]
        if len(d) != 1:
            print("Unclear which model to evaluate.")
            print("Found ", d)
            return -1
        model_file = os.path.join(base_dir, str(split), d[0], "model_" + str(model_no) + ".pth")
        if not os.path.exists(model_file):
            print("Cannot read model " + model_file)
            return -1
        evaluate_model(args, model_file, splits, split, gesture_ids, logits_per_video, labels_per_video,
                       tb_writer=tb_writer)
        # TODO
        # break

    torch.save(logits_per_video, logits_file)
    if labels_file is not None:
        torch.save(labels_per_video, labels_file)


def evaluate_model(args, model_file, splits, split, gesture_ids, logits_per_video, labels_per_video=None,
                   tb_writer=None):

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Test model " + model_file)
    model_weights = torch.load(model_file)

    num_class = len(gesture_ids)

    net = GestureClassifier(num_class, dropout=0.0, snippet_length=args.snippet_length, input_size=args.input_size)
    net.load_state_dict(model_weights, strict=True)
    # === load data... ===

    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    test_lists = splits[split:split + 1]
    test_lists = list(map(lambda x: os.path.join(lists_dir, x), test_lists))

    normalize = GroupNormalize(net.input_mean, net.input_std)
    test_augmentation = torchvision.transforms.Compose([GroupScale(int(net.scale_size)),
                                                        GroupCenterCrop(net.crop_size)])

    test_videos = list()
    for list_file in test_lists:
        test_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
    test_loaders = list()

    for video in test_videos:
        data_set = SequentialGestureDataSet(args.data_path, args.transcriptions_dir, gesture_ids,
                                            video[0], int(video[1]), snippet_length=net.snippet_length,
                                            kinematics_dir=args.kinematics_dir,
                                            video_sampling_step=args.video_sampling_step,
                                            snippet_sampling_step=args.snippet_sampling_step,
                                            modality=args.modality, image_tmpl=args.image_tmpl,
                                            video_suffix=args.video_suffix, return_3D_tensor=net.is_3D_architecture,
                                            transform=test_augmentation, normalize=normalize,
                                            load_to_RAM=False, transpose_img=False)
        test_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.workers))

    # === evaluate model... ===

    net = net.to(device_gpu)
    net.eval()
    count = 1
    hidden_features = None
    labels_all = np.array([], dtype=np.int64)
    with torch.no_grad():
        for test_loader in test_loaders:

            video_id = test_loader.dataset.video_id
            print("Evaluate video " + video_id)

            logits = None
            labels = None

            if labels_per_video is not None:
                labels = np.array([], dtype=np.int64)
            for _, batch in enumerate(test_loader):
                img_data, kin_data, target = batch
                img_data = img_data.to(device_gpu)
                kin_data = kin_data.to(device_gpu)

                if args.return_feature:
                    out, feature = net(img_data, kin_data, return_feature=True)
                    feature = feature.to(device_cpu)
                    if hidden_features is None:
                        hidden_features = feature
                    else:
                        hidden_features = torch.cat((hidden_features, feature), dim=0)
                    labels_all = np.append(labels_all, target.numpy())
                else:
                    out = net(img_data)

                if logits is None:
                    logits = out.to(device_cpu)
                else:
                    logits = torch.cat((logits, out.to(device_cpu)), dim=0)
                if labels is not None:
                    labels = np.append(labels, target.numpy())


            # TODO !!
            # if args.return_feature:
            #     if count != 1:
            #         count = count+1
            #         pass
            #     else:
            #         _data, _label = select_n_random(hidden_features.numpy(), labels_all, 100)
            #         tb_writer.add_embedding(_data, metadata=_label)
            #         tb_writer.close()
            #         exit()

            logits_per_video[video_id] = logits
            if labels_per_video is not None:
                labels_per_video[video_id] = labels


if __name__ == '__main__':
    args = parser.parse_args()
    args.video_suffix = "_capture2"
    args.image_tmpl = '{:d}.png'
    if args.modality == 'Flow':
        args.image_tmpl = 'flow_{}_{:05d}.jpg'

    if args.data_path == '?':
        print("Please specify the path to your image data using the --data_path option or set an appropriate default "
              "in test_opts.py!")
    else:
        if args.transcriptions_dir == '?':
            print("Please specify the path to the transcription files using the --transcriptions_dir option or set "
                  "an appropriate default in test_opts.py!")
        else:
            if args.model_dir == '?':
                print("Please specify the path to your model folder using the --model_dir option or set an appropriate "
                      "default in test_opts.py!")
            else:
                if args.kinematics_dir == '?':
                    print("Please specify the path to the kinematics files using the --kinematics_dir option or set "
                          "an appropriate default in train_opts.py!")
                else:
                    eval_exp(args)
