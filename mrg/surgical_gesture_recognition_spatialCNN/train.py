# Training file

from train_opts import parser
from models import GestureClassifier
from dataset import GestureDataSet, SequentialGestureDataSet
from transforms import GroupNormalize, GroupScale, GroupCenterCrop
from metrics import accuracy, average_F1, edit_score, overlap_f1
from util import AverageMeter, splits_LOSO, splits_LOUO, splits_LOUO_NP, gestures_SU, gestures_NP, gestures_KT
import util

import os.path
import datetime
import numpy as np
import string
import torch
import torchvision


def main(args):
    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return

    device_gpu = torch.device("cuda:0")
    device_cpu = torch.device("cpu")

    checkpoint = None
    if args.resume_exp:
        output_folder = args.resume_exp
    else:
        output_folder = os.path.join(args.out, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
                                     args.eval_scheme, str(args.split), datetime.datetime.now().strftime("%H%M"))
        os.makedirs(output_folder)

    f_log = open(os.path.join(output_folder, "log.txt"), "a")
    def log(msg):
        util.log(f_log, msg)
    checkpoint_file = os.path.join(output_folder, "checkpoint" + ".pth.tar")

    if args.resume_exp:
        checkpoint = torch.load(checkpoint_file)
        args_checkpoint = checkpoint['args']
        for arg in args_checkpoint:
            setattr(args, arg, args_checkpoint[arg])
        log("====================================================================")
        log("Resuming experiment...")
        log("====================================================================")
    else:
        if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
            args.data_path = args.data_path.format(args.task)
        if len([t for t in string.Formatter().parse(args.video_lists_dir)]) > 1:
            args.video_lists_dir = args.video_lists_dir.format(args.task)
        if len([t for t in string.Formatter().parse(args.transcriptions_dir)]) > 1:
            args.transcriptions_dir = args.transcriptions_dir.format(args.task)

        log("Used parameters...")
        for arg in sorted(vars(args)):
            log("\t" + str(arg) + " : " + str(getattr(args, arg)))

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if checkpoint:
        torch.set_rng_state(checkpoint['rng'])

    # ===== prepare model =====

    gesture_ids = None
    if args.task == "Suturing":
        gesture_ids = gestures_SU
    elif args.task == "Needle_Passing":
        gesture_ids = gestures_NP
    elif args.task == "Knot_Tying":
        gesture_ids = gestures_KT
    num_class = len(gesture_ids)

    model = GestureClassifier(num_class, dropout=args.dropout,
                                 snippet_length=args.snippet_length, input_size=args.input_size)

    if checkpoint:
        # load model weights
        model.load_state_dict(checkpoint['model_weights'])

    log("param count: {}".format(sum(p.numel() for p in model.parameters())))
    log("trainable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.loss_weighting is True:
        loss_weights = list(range(1, args.snippet_length + 1))
        loss_weights = [x ** 2 for x in loss_weights]
        w_sum = np.sum(np.array(loss_weights))
        loss_weights = [x / w_sum for x in loss_weights]
        loss_weights = torch.from_numpy(np.array(loss_weights))
        loss_weights = loss_weights.to(device_gpu, dtype=torch.double)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([
        {'params': model.base_model.parameters()},
        {'params': model.fc_h.parameters()},
        {'params': model.fc.parameters()},
    ], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if checkpoint:
        # load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_gpu)
    scheduler = None
    if args.use_scheduler:
        last_epoch = -1
        if checkpoint:
            last_epoch = checkpoint['epoch']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step,
                                                    gamma=args.scheduler_gamma, last_epoch=last_epoch)

    # ===== load data =====

    splits = None
    if args.eval_scheme == 'LOSO':
        splits = splits_LOSO
    elif args.eval_scheme == 'LOUO':
        if args.task == "Needle_Passing":
            splits = splits_LOUO_NP
        else:
            splits = splits_LOUO
    assert (args.split >= 0 and args.split < len(splits))
    train_lists = splits[0:args.split] + splits[args.split + 1:]
    test_lists = []
    test_lists.append(str(splits[args.split]))

    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    train_lists = list(map(lambda x: os.path.join(lists_dir, x), train_lists))
    test_lists = list(map(lambda x: os.path.join(lists_dir, x), test_lists))
    log("Splits in train set :" + str(train_lists))
    log("Splits in test set :" + str(test_lists))

    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(crop_corners=args.corner_cropping,
                                                do_horizontal_flip=args.do_horizontal_flip)

    train_set = GestureDataSet(args.data_path, train_lists, args.transcriptions_dir, gesture_ids, split_set=args.split,
                               kinematics_dir=args.kinematics_dir, vision_dir=args.vision_dir, snippet_length=model.snippet_length,
                               min_overlap=1, video_sampling_step=args.video_sampling_step, modality=args.modality,
                               image_tmpl=args.image_tmpl, video_suffix=args.video_suffix,
                               return_3D_tensor=model.is_3D_architecture, return_dense_labels=True,
                               transform=train_augmentation, normalize=normalize,
                               load_to_RAM=args.data_preloading, transpose_img=False)

    def init_train_loader_worker(worker_id):
        np.random.seed(int((torch.initial_seed() + worker_id) % (2**32)))  # account for randomness
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, worker_init_fn=init_train_loader_worker)
    log("Training set: will sample {} gesture snippets per pass".format(train_loader.dataset.__len__()))

    val_augmentation = torchvision.transforms.Compose([GroupScale(int(model.scale_size)),
                                                       GroupCenterCrop(model.crop_size)])
    test_videos = list()
    for list_file in test_lists:
        test_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
    val_loaders = list()
    for video in test_videos:
        data_set = SequentialGestureDataSet(args.data_path, args.transcriptions_dir, gesture_ids, video[0], int(video[1]),
                                            kinematics_dir=args.kinematics_dir,
                                            snippet_length=model.snippet_length,
                                            video_sampling_step=args.video_sampling_step,
                                            snippet_sampling_step=args.snippet_sampling_step,
                                            modality=args.modality, image_tmpl=args.image_tmpl,
                                            video_suffix=args.video_suffix, return_3D_tensor=model.is_3D_architecture,
                                            transform=val_augmentation, normalize=normalize,
                                            load_to_RAM=args.data_preloading, transpose_img=False)
        val_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
                                                       shuffle=False, num_workers=args.workers))

    log("Validation set: ")
    for val_loader in val_loaders:
        log("{} ({})".format(val_loader.dataset.video_id, val_loader.dataset.__len__()))

    # ===== train model =====
    best_acc = 0.0
    best_epoch = 0

    log("Start training...")

    model = model.to(device_gpu)
    model = torch.nn.DataParallel(model)
    # torch.autograd.set_detect_anomaly(True)

    start_epoch = 0
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['best_epoch']
    for epoch in range(start_epoch, args.epochs):

        ctr = 0
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        model.train()
        while ctr < args.steps_per_epoch:
            for _, batch in enumerate(train_loader):

                optimizer.zero_grad()

                img_data, kin_data, target = batch
                batch_size = target.size(0)
                img_data = img_data.to(device_gpu)
                kin_data = kin_data.to(device_gpu)

                target = target.to(device_gpu, dtype=torch.int64)

                output = model(img_data)
                target = target[:, -1]
                # print(img_data.shape)
                # print(output.shape)
                # print(target.shape)
                # torch.Size([32, 1, 3, 224, 224])
                # torch.Size([32, 10])
                # torch.Size([32])

                if len(target.shape) <= 2 or not args.loss_weighting:
                    # target = target.squeeze(1)
                    loss = criterion(output, target)
                else:
                    loss = torch.zeros(batch_size).to(device_gpu)
                    for i in range(batch_size):
                        t_i = target[i, :]
                        o_i = output[i, :]
                        o_i = o_i.permute(1, 0)
                        l_i = torch.nn.CrossEntropyLoss(reduction='none')(o_i, t_i).to(torch.double)
                        l_i = torch.sum(torch.mul(l_i, loss_weights))
                        loss[i] = l_i
                    loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), batch_size)

                if len(output.shape) > 2:
                    output = output[:, :, -1]  # consider only final prediction
                    target = target[:, -1]
                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)
                acc = (predicted == target).sum().item() / batch_size
                train_acc.update(acc, batch_size)
                ctr += batch_size

        log("Epoch {}: Train loss: {train_loss.avg:.4f} Train acc: {train_acc.avg:.3f}"
            .format(epoch, train_loss=train_loss, train_acc=train_acc))

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:

            model.eval()

            with torch.no_grad():

                overall_acc = []
                overall_avg_f1 = []
                overall_edit = []
                overall_f1_10 = []
                overall_f1_25 = []
                overall_f1_50 = []
                overall_loss = []
                for val_loader in val_loaders:
                    P = np.array([], dtype=np.int64)
                    Y = np.array([], dtype=np.int64)
                    for _, batch in enumerate(val_loader):
                        img_data, kin_data, target = batch
                        Y = np.append(Y, target.numpy())

                        img_data = img_data.to(device_gpu)
                        kin_data = kin_data.to(device_gpu)

                        output = model(img_data)
                        # print(img_data.shape)
                        # print(target.shape)
                        # print(output.shape)
                        # exit()
                        # target.shape [64]
                        # output.shape [64, 10, 16]

                        if len(output.shape) > 2:
                            output = output[:, :, -1]  # consider only final prediction
                        predicted = torch.nn.Softmax(dim=1)(output)
                        _, predicted = torch.max(predicted, 1)
                        P = np.append(P, predicted.to(device_cpu).numpy())
                        val_loss = criterion(output.cpu(), target)

                    acc = accuracy(P, Y)
                    avg_f1, _ = average_F1(P, Y, n_classes=num_class)
                    edit = edit_score(P, Y)
                    f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
                    f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
                    f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)
                    # log("Trial {}:\tAcc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f}"
                    #     .format(val_loader.dataset.video_id, acc, avg_f1, edit, f1_10, f1_25, f1_50))

                    overall_acc.append(acc)
                    overall_avg_f1.append(avg_f1)
                    overall_edit.append(edit)
                    overall_f1_10.append(f1_10)
                    overall_f1_25.append(f1_25)
                    overall_f1_50.append(f1_50)
                    overall_loss.append(val_loss)

                log("Overall: Acc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f} \
                Loss {:.4f}".format(
                    np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit),
                    np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50),
                    np.mean(overall_loss)
                ))

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            # ===== save model =====
            model_file = os.path.join(output_folder, "model_" + str(epoch) + ".pth")
            torch.save(model.module.state_dict(), model_file)
            log("Saved model to " + model_file)

        if np.mean(overall_acc) > best_acc:
            best_acc = np.mean(overall_acc)
            best_epoch = epoch
            model_file = os.path.join(output_folder, "model_-1" + ".pth")
            torch.save(model.module.state_dict(), model_file)
            log("Best model epoch/acc: " + str(best_epoch)+"/"+format(best_acc, '.3f'))

        # ===== save checkpoint =====
        current_state = {'epoch': epoch + 1,
                         'model_weights': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'rng': torch.get_rng_state(),
                         'args': args_dict,
                         'best_acc': best_acc,
                         'best_epoch': best_epoch
                         }
        torch.save(current_state, checkpoint_file)

    f_log.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args.video_suffix = "_capture2"
    args.image_tmpl = '{:d}.png'
    if args.modality == 'Flow':
        args.image_tmpl = 'flow_{}_{:05d}.jpg'
    args.eval_batch_size = 2 * args.batch_size

    if args.data_path == '?':
        print("Please specify the path to your image data using the --data_path option or set an appropriate default "
              "in train_opts.py!")
    else:
        if args.transcriptions_dir == '?':
            print("Please specify the path to the transcription files using the --transcriptions_dir option or set "
                  "an appropriate default in train_opts.py!")
        else:

            if args.kinematics_dir == '?':
                print("Please specify the path to the kinematics/vision_features files using the "
                      "--kinematics_dir option or set "
                      "an appropriate default in train_opts.py!")
            else:
                if args.out == '?':
                    print("Please specify the path to your output folder using the --out option or set an appropriate "
                          "default in train_opts.py!")
                else:
                    main(args)
