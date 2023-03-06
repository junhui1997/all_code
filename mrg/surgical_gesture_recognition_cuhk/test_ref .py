from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch 
import torch.nn as nn
from random import randrange
from tcn_model_orii import TcnGcnNet
from my_dataset import RawFeatureDataset
from config import args
import copy

from logger import Logger
import utils
import pdb
from sklearn import metrics

from config import (raw_feature_dir, sample_rate, graph_dir,
                    gesture_class_num, dataset_name)
from config import tcn_params, tcn_run_num, result_dir, dataset_name
from config import args
from torch.utils.tensorboard import SummaryWriter
from utils import gestures_MISAW

# writer = SummaryWriter('visualize/test')

def test_model(model, test_dataset, loss_weights=None, plot_naming=None):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1, shuffle=False)
    model.eval()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).cuda(),
                        ignore_index=-1)

    #Test the Model
    total_loss = 0
    preditions = []
    pred_score = []
    gts=[]

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            feature = data['feature'].float()
            feature = feature.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.cuda()

            kinematics = data['kinematics'].float()
            kinematics = kinematics.cuda()

            file_name = data['name'][0]
            print("Evaluating....", file_name)

            # Forward
            out = model(feature, kinematics)
            out = out.squeeze(0)

            loss = criterion(input=out, target=gesture)

            total_loss += loss.item()

            out = torch.nn.functional.softmax(out,dim=1)
            score, pred = out.data.max(1)

            trail_len = data['trail_len']
            gesture = gesture[:trail_len]
            pred = pred[:trail_len]
            score = score[:trail_len]
            # emb = emb[:trail_len]

            preditions.append(pred.cpu().numpy())
            gts.append(gesture.data.cpu().numpy())
            pred_score.append(score.cpu().numpy())

            # Plot  
            if plot_naming:    
                graph_file = os.path.join(graph_dir, '{}_seq_{}'.format(
                                                plot_naming, file_name))

                utils.plot_barcode(gt=None, 
                                   pred=pred.cpu().numpy(), 
                                   visited_pos=None,
                                   show=False, save_file=graph_file)
            
            # writer.add_embedding(emb.data.cpu().numpy(), metadata=m_label)
            save_root = './temp_submission/Step/'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_pred_name = file_name + "_Results_Step.txt"
            save_score_name = file_name + "_Results_Step_Score.txt"
            
            # save prediction
            save_pred_str = []
            downsample_rate = 3
            for i, _label in enumerate(pred):
                save_pred_str.append([str(i*3), gestures_MISAW[int(_label)]])

            save_pred_str = np.array(save_pred_str)

            np.savetxt(save_root+save_pred_name, save_pred_str,fmt='%s\t%s')

            save_score_str = []
            for i, _score in enumerate(score):
                save_score_str.append([str(i*3), str(float(_score))])

            save_score_str = np.array(save_score_str)

            np.savetxt(save_root+save_score_name, save_score_str, fmt='%s\t%s')


    model.train()
    return


######################### Main Process #########################

def test_(model_params, train_params, naming):

    # Get trail list
    cross_val_splits = utils.get_cross_val_splits()

    # Cross-Validation Result
    result = []
    gts_all = []
    preds_all = []

    # Cross Validation
    for split_idx, split in enumerate(cross_val_splits):

        feature_dir = os.path.join(raw_feature_dir, split['name'])
        test_trail_list = split['test']
        train_trail_list = split['train']

        split_naming = naming + '_split_{}'.format(-1)

        trained_model_file = utils.get_tcn_model_file(split_naming)
        log_dir = utils.get_tcn_log_sub_dir(split_naming)

        # Model
        model = TcnGcnNet(**model_params)
        model = model.cuda()

        print(model)

        n_layers = len(model_params['encoder_params']['layer_sizes'])

        # Dataset
        train_dataset = RawFeatureDataset(dataset_name, 
                                          feature_dir,
                                          train_trail_list,
                                          encode_level=n_layers,
                                          sample_rate=sample_rate,
                                          sample_aug=False,
                                          normalization=None)

        test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
        test_dataset = RawFeatureDataset(dataset_name, 
                                         feature_dir,
                                         test_trail_list,
                                         encode_level=n_layers,
                                         sample_rate=sample_rate,
                                         sample_aug=False,
                                         normalization=test_norm)

        model.load_state_dict(torch.load(trained_model_file))

        test_model(model, test_dataset, 
                    loss_weights=None,
                    plot_naming=split_naming)


    return


def main():
    naming = 'run_{}_{}'.format((1), args.save_name)

    test_(tcn_params['model_params'], tcn_params['train_params'], naming)  #8x6



if __name__ == '__main__':
    main()
