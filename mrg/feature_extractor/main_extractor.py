import argparse
import torch
import torch.nn as nn
from my_dataset import RawFeatureDataset
import warnings
import utils
from config import args
from models import cnn_feature18
import torch.nn.functional as F
import numpy as np
import time
import torch.optim.lr_scheduler as lr_scheduler
warnings.filterwarnings('ignore')
import os
from config import (raw_feature_dir, sample_rate, graph_dir,
                    gesture_class_num, dataset_name)


def _process_one_batch(batch_x,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_x = batch_x.float()
    batch_x = batch_x.to(device)

    outputs = model(batch_x)

    return outputs

def vali(vali_loader, criterion, model):
    #首先设置的是eval模式
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss = []
    correct, total = 0, 0
    for i, (batch_x, batch_y, batch_z) in enumerate(vali_loader):
        pred = _process_one_batch(batch_x, model)
        true = batch_y.to(device)
        loss = criterion(pred.detach().cpu(), true.detach().cpu())
        total_loss.append(loss)
        # calculate accy
        pred_idx = F.log_softmax(pred, dim=1).argmax(dim=1)
        total += true.size(0)  # 统计了batch_size
        correct += (pred_idx == true).sum().item()

    total_loss = np.average(total_loss)
    acc = correct / total
    model.train()
    return total_loss, acc
def train(model, args, train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.cuda()
    model_optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=2)
    criterion = nn.CrossEntropyLoss()
    train_steps = len(train_loader)
    best_accuracy = -1
    time_now = time.time()
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_z) in enumerate(train_loader):
            iter_count += 1

            # 需要每一个batch时候zero_grad！
            model_optim.zero_grad()

            pred = _process_one_batch(batch_x, model)
            true = batch_y.to(device)
            loss = criterion(pred, true)
            train_loss.append(loss.item())

            # 每100个iter重新更新一次预估时间
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                # 从开始到现在的时间/经历了多少iteration
                speed = (time.time() - time_now) / iter_count
                # train_step反映了一个epoch有多少个iter
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        test_loss, current_accuray = vali(test_loader, criterion, model)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, 0, test_loss))

        print('current accuracy in Epoch: {0} is {1:.7f}'.format(epoch + 1, current_accuray))
        if current_accuray > best_accuracy:
            path = 'checkpoints'
            best_accuracy = current_accuray
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
            best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print('best accuracy is ', best_accuracy)




def main():
    cross_val_splits = utils.get_cross_val_splits(args)
    for split_idx, split in enumerate(cross_val_splits):
        # TODO !!!
        # 这里可以看到是只跑了想要的那个split里面的数据，其他的数据没有管
        if (split_idx+1) != args.split:
            continue
        feature_dir = os.path.join(raw_feature_dir, split['name'])
        test_trail_list = split['test']
        train_trail_list = split['train']

    #
    train_dataset = RawFeatureDataset(args, train_trail_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32, shuffle=True)
    test_enc = train_dataset.get_enc()
    test_dataset = RawFeatureDataset(args, test_trail_list, test_enc, flag='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=32, shuffle=True)

    model = cnn_feature18(args=args)
    train(model, args, train_loader, test_loader)

if __name__ == '__main__':
    main()
