from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch 
import torch.nn as nn
from random import randrange
from tcn_model import TcnGcnNet
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

    m_preditions = []
    m_gts=[]
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
            out= model(feature, kinematics)
            out = out.squeeze(0)

            loss = criterion(input=out, target=gesture)

            total_loss += loss.item()

            out = torch.nn.functional.softmax(out,dim=1)
            score, pred = out.data.max(1)

            trail_len = (gesture.data.cpu().numpy()!=-1).sum()
            gesture = gesture[:trail_len]
            pred = pred[:trail_len]
            score = score[:trail_len]

            preditions.append(pred.cpu().numpy())
            gts.append(gesture.data.cpu().numpy())
            pred_score.append(score.cpu().numpy())
            m_preditions.extend(pred.cpu().numpy())
            m_gts.extend(gesture.data.cpu().numpy())

            # Plot  
            if plot_naming:    
                graph_file = os.path.join(graph_dir, '{}_seq_{}'.format(
                                                plot_naming, str(i)))

                utils.plot_barcode(gt=gesture.data.cpu().numpy(), 
                                   pred=pred.cpu().numpy(), 
                                   visited_pos=None,
                                   show=False, save_file=graph_file)
            
            save_root = './temp_submission/Step/'
            save_score_root = './temp_submission/Step_score/'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            if not os.path.exists(save_score_root):
                os.makedirs(save_score_root)
            save_pred_name = file_name + "_Results_Step.txt"
            save_score_name = file_name + "_Results_Step.txt"
            
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

            np.savetxt(save_score_root+save_score_name, save_score_str, fmt='%s\t%s')

    bg_class = 0 if dataset_name != 'JIGSAWS' else None

    avg_loss = total_loss / len(test_loader.dataset)
    edit_score = utils.get_edit_score_colin(preditions, gts,
                                            bg_class=bg_class)
    accuracy = utils.get_accuracy_colin(preditions, gts)
    #accuracy = utils.get_accuracy(preditions, gts)
    
    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                        n_classes=gesture_class_num, 
                                        bg_class=bg_class, 
                                        overlap=overlap))

    model.train()
    return accuracy, edit_score, avg_loss, f_scores, m_gts, m_preditions


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
        # TODO !!!
        if (split_idx+1) != args.split:
           continue
        feature_dir = os.path.join(raw_feature_dir, split['name'])
        test_trail_list = split['test']
        train_trail_list = split['train']

        split_naming = naming + '_split_{}'.format(split_idx+1)

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

        acc, edit, _, f_scores, gts, preds = test_model(model, test_dataset, 
                                            loss_weights=None,
                                            plot_naming=split_naming)

        result.append([acc, edit, f_scores[0], f_scores[1], 
                                  f_scores[2], f_scores[3]])
        gts_all.extend(gts)
        preds_all.extend(preds)

        print('Acc: ', acc)
        print('Edit: ', edit)
        print('F10: ', f_scores[0])
        print('F25: ', f_scores[1])
        print('F50: ', f_scores[2])
        print('F75: ', f_scores[3])

    result = np.array(result)
    gts_all = np.array(gts_all)
    preds_all = np.array(preds_all)

    return result, gts_all, preds_all


def main():
    naming = 'run_{}_{}'.format((1), args.save_name)

    run_result, gts_all, preds_all = test_(tcn_params['model_params'], 
                                    tcn_params['train_params'],
                                    naming)  #8x6

    # classify_report = metrics.classification_report(gts_all, preds_all)
    # # np.save("./acc_RGCN.npy",classify_report)
    # print(classify_report)
    # exit()


if __name__ == '__main__':
    main()
