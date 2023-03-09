import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler

#只有type4时候需要用到val_loss这个参数，其他几个都是设定好得直接往下走
def adjust_learning_rate(optimizer, epoch, args , scheduler = None,val_loss=0):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        base_lr = args.learning_rate
        # 这个地方/2代表了整个周期内会执行两次波形 ，eta_min代表的是最小值会到多少
        sched = cosine(t_max=args.train_epochs / 2, eta_min=base_lr / 100)
        lr_adjust = {
            i: sched(i, base_lr) for i in range(args.train_epochs)
        }
    elif args.lradj == 'type4':
        lr_prev = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        if lr != lr_prev:
            print('Updating learning rate to {}'.format(lr))
        return

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        #这里的o是dim，压缩掉了第一个dim
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class StandardScaler_classification():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, df, col_name):
        all_array = None
        self.col_name = col_name
        for i in range(len(df[self.col_name])):
            if all_array is None:
                all_array = df[self.col_name].iloc[i]
            else:
                all_array = np.append(all_array, df[self.col_name].iloc[i], axis=0)
        self.mean = all_array.mean(0)
        self.std = all_array.std(0)

    def transform(self, df):
        df[self.col_name] = df[self.col_name].apply(lambda x: (x - self.mean) / self.std)
        return df


# 在同一次程序里面get_fold返回的df是一致的，避免了信息泄漏
def get_fold(df_kfold,folds,key_name):
    random_state = int(np.sqrt(torch.initial_seed()))
    kfolder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    df_kfold[key_name] = df_kfold[key_name].apply(str)
    val_indices = [val_indices for _, val_indices in kfolder.split(df_kfold[key_name], df_kfold[key_name])]
    df_kfold['fold'] = -1
    #给每个都打上了fold index
    for i, vi in enumerate(val_indices):
        #print(i,vi)
        folder_idx = df_kfold.index[vi]
        df_kfold.loc[folder_idx,'fold'] = i
    return df_kfold