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
import torch.optim.lr_scheduler as lr_scheduler
from module_box.edit_score_dist import calculate_edit_score
from module_box.edit_score_dist import flatten_list
from config import (raw_feature_dir, sample_rate, graph_dir,
                    gesture_class_num, dataset_name)


def train_model(model,
                train_dataset,
                val_dataset,
                num_epochs,
                learning_rate,
                batch_size,
                weight_decay,
                loss_weights=None,
                trained_model_file=None,
                log_dir=None):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size, shuffle=True)
    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).cuda(),
                        ignore_index=-1)

    # Logger
    if log_dir is not None:
        logger = Logger(log_dir) 

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=200)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
    #                             momentum=0.9, weight_decay=1e-4)

    step = 1
    update_bs = 8
    v_accuracy_best = 0.0
    v_edit_score_best = 0.0
    best_model = None
    for epoch in range(num_epochs):
        print(epoch)
        batch_loss = 0.0
        for i, data in enumerate(train_loader):
            # [batch_size,padded_video_len, dim] for video dim = 128 for kine dim = 14
            feature = data['feature'].float()
            feature = feature.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.cuda()

            kinematics = data['kinematics'].float()
            kinematics = kinematics.cuda()

            # Forward
            # out (padded_video_len,num_class)
            # ????????????????????????loss??????????????????????????????????????????
            out = model(feature, kinematics)
            flatten_out = out.view(-1, out.shape[-1])

            loss = criterion(input=flatten_out, target=gesture)

            # a sample update=====================
            # Backward and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # Logging
            # if log_dir is not None:
            #     logger.scalar_summary('loss', loss.item(), step)

            # step += 1
            # ==================================


            # batch update=====================
            # Backward and optimize
            if step % update_bs == 1:
                optimizer.zero_grad()

            loss_2 = loss / update_bs
            loss_2.backward()
            
            batch_loss = batch_loss + loss_2.item()
            
            if step % update_bs == 0:
                optimizer.step()
                # Logging
                if log_dir is not None:
                    logger.scalar_summary('loss', batch_loss, step)
            
                batch_loss = 0.0
            step += 1
            # ==================================


        if log_dir is not None:
            train_result = test_model(model, train_dataset, loss_weights)
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            val_result = test_model(model, val_dataset, loss_weights)
            v_accuracy, v_edit_score, v_loss, v_f_scores = val_result


            lr_prev = optimizer.param_groups[0]['lr']
            scheduler.step(v_loss)
            lr = optimizer.param_groups[0]['lr']
            if lr != lr_prev:
                print('Updating learning rate to {}'.format(lr))
            print('train_accuracy:{}, edit_score:{}, train_loss:{}'.format(t_accuracy, t_edit_score, t_loss))
            print('val_accuracy:{}, edit_score:{}, val_loss:{}'.format(v_accuracy,v_edit_score,v_loss))

            logger.scalar_summary('t_accuracy', t_accuracy, epoch)
            logger.scalar_summary('t_edit_score', t_edit_score, epoch)
            logger.scalar_summary('t_loss', t_loss, epoch)
            logger.scalar_summary('t_f_scores_10', t_f_scores[0], epoch)
            logger.scalar_summary('t_f_scores_25', t_f_scores[1], epoch)
            logger.scalar_summary('t_f_scores_50', t_f_scores[2], epoch)
            logger.scalar_summary('t_f_scores_75', t_f_scores[3], epoch)

            logger.scalar_summary('v_accuracy', v_accuracy, epoch)
            logger.scalar_summary('v_edit_score', v_edit_score, epoch)
            logger.scalar_summary('v_loss', v_loss, epoch)
            logger.scalar_summary('v_f_scores_10', v_f_scores[0], epoch)
            logger.scalar_summary('v_f_scores_25', v_f_scores[1], epoch)
            logger.scalar_summary('v_f_scores_50', v_f_scores[2], epoch)
            logger.scalar_summary('v_f_scores_75', v_f_scores[3], epoch)

            # TODO!!!
            if v_accuracy > v_accuracy_best:
                best_model = copy.deepcopy(model)
                v_accuracy_best = v_accuracy

        if trained_model_file is not None:
            torch.save(best_model.state_dict(), trained_model_file)


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

            # Forward
            out = model(feature, kinematics)
            out = out.squeeze(0)

            loss = criterion(input=out, target=gesture)

            total_loss += loss.item()

            pred = out.data.max(1)[1]

            trail_len = (gesture.data.cpu().numpy()!=-1).sum()
            gesture = gesture[:trail_len]
            pred = pred[:trail_len]

            preditions.append(pred.cpu().numpy())
            gts.append(gesture.data.cpu().numpy())

            # Plot  
            if plot_naming:    
                graph_file = os.path.join(graph_dir, '{}_seq_{}'.format(
                                                plot_naming, str(i)))

                utils.plot_barcode(gt=gesture.data.cpu().numpy(), 
                                   pred=pred.cpu().numpy(), 
                                   visited_pos=None,
                                   show=False, save_file=graph_file)

    bg_class = 0 if dataset_name != 'JIGSAWS' else None

    avg_loss = total_loss / len(test_loader.dataset)
    edit_score = utils.get_edit_score_colin(preditions, gts,
                                            bg_class=bg_class)

    # es = 0
    # ed = 0
    # es_l = []
    # for i in range(len(preditions)):
    #     edit_distance, edit_score_h = calculate_edit_score(preditions[i], gts[i])
    #     es_l.append(es)
    #     es+=edit_score_h
    #     ed+=edit_distance
    # #print(es_l)
    # ed = ed/len(preditions)
    # es = es/len((preditions))

    #print(edit_score, es, ed, len(flatten_list(preditions)))
    accuracy = utils.get_accuracy_colin(preditions, gts)
    #accuracy = utils.get_accuracy(preditions, gts)
    
    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                        n_classes=gesture_class_num, 
                                        bg_class=bg_class, 
                                        overlap=overlap))

    model.train()
    return accuracy, edit_score, avg_loss, f_scores


######################### Main Process #########################

def cross_validate(model_params, train_params, naming):

    # Get trail list
    cross_val_splits = utils.get_cross_val_splits()

    # Cross-Validation Result
    result = []

    # Cross Validation
    for split_idx, split in enumerate(cross_val_splits):
        # TODO !!!
        # ?????????????????????????????????????????????split??????????????????????????????????????????
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

        loss_weights = utils.get_class_weights(train_dataset)
        #loss_weights = None

        if train_params is not None:
            train_model(model, 
                        train_dataset,
                        test_dataset, 
                        **train_params,
                        loss_weights=loss_weights,
                        trained_model_file=trained_model_file,
                        log_dir=log_dir)
                        #log_dir=None)

        model.load_state_dict(torch.load(trained_model_file))

        acc, edit, _, f_scores = test_model(model, test_dataset, 
                                            loss_weights=loss_weights,
                                            plot_naming=split_naming)

        result.append([acc, edit, f_scores[0], f_scores[1], 
                                  f_scores[2], f_scores[3]])

        print('Acc: ', acc)
        print('Edit: ', edit)
        print('F10: ', f_scores[0])
        print('F25: ', f_scores[1])
        print('F50: ', f_scores[2])
        print('F75: ', f_scores[3])

    result = np.array(result)

    return result
