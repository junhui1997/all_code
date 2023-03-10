import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import scipy.io
import torch


def visualize_gesture_predictions(out_dir, model_dir, exps_to_compare, path_to_colins_result, exp_descriptions,
                                  sequence_to_visualize, exps_to_evaluate, eval_scheme, eval_freq, model_no):

    # calculate average recognition accuracy for each video to determine which video (-->gesture sequence) to visualize
    metric = 'acc'
    eval_type = 'plain'
    avg_exp_results = {}
    for exp in exps_to_evaluate:
        eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), eval_type, exp,
                                 "{}.pth.tar".format(model_no))
        eval_results = torch.load(eval_file)
        for key in eval_results:
            if key == 'overall':
                pass
            else:  # key = video_id
                results_per_video = eval_results[key]
                if key not in avg_exp_results:
                    avg_exp_results[key] = []
                avg_exp_results[key].append(results_per_video[metric])
    for video_id in avg_exp_results:
        avg_exp_results[video_id] = np.mean(avg_exp_results[video_id])
    avg_exp_results = [(video_id, avg_accuracy) for video_id, avg_accuracy in avg_exp_results.items()]
    avg_exp_results = sorted(avg_exp_results, key=lambda x: x[1])
    if sequence_to_visualize == "lowest":
        sequence = avg_exp_results[0][0]
    elif sequence_to_visualize == "highest":
        sequence = avg_exp_results[-1][0]
    elif sequence_to_visualize == "median":
        sequence = avg_exp_results[len(avg_exp_results) // 2][0]
    else:
        print("Unclear which sequence to visualize. Should be one of ['lowest', 'median', 'highest']")
        return

    sequences_to_plot = []
    for exp in exps_to_compare:
        eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), exp,
                                 "{}.pth.tar".format(model_no))
        eval_results = torch.load(eval_file)
        sequences_to_plot.append(eval_results[sequence]['P'])

    if path_to_colins_result:  # find results reproduced from Colin Lea et al.
        data_splits = {'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}
        sequence_id = sequence.split('_')[-1]  # e.g. "F002"
        user_id = sequence_id[0]
        trial_no = int(sequence_id[1:])
        mat = scipy.io.loadmat(os.path.join(path_to_colins_result, "Split_{}.mat".format(data_splits[user_id])))
        split_results = mat['P'].squeeze()
        trial_result = split_results[trial_no - 1].squeeze()
        sequences_to_plot.append(trial_result)

    # add ground truth
    eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), exps_to_compare[0],
                             "{}.pth.tar".format(model_no))
    eval_results = torch.load(eval_file)
    sequences_to_plot.append(eval_results[sequence]['Y'])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, sequence_to_visualize + ".svg")

    fig, axes = _plot_label_sequences(sequences_to_plot, exp_descriptions, num_classes=10)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def _plot_label_sequences(label_seqs, seq_names, num_classes, fig_width=8, fig_height_per_seq=0.4):
    max_seq_length = 2200
    text_offset = 50

    num_seqs = len(label_seqs)
    figsize = (fig_width, num_seqs * fig_height_per_seq)
    fig, axes = plt.subplots(nrows=num_seqs, ncols=1, sharex=True, figsize=figsize,
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    for label_seq, name, ax in zip(label_seqs, seq_names, axes):
        plt.sca(ax)
        plt.axis('off')
        x = np.arange(0, label_seq.size)
        y = 0 * np.ones(label_seq.size)
        ax.scatter(x, y, c=label_seq, marker='|', s=300, lw=2, vmin=0, vmax=num_classes, cmap='tab10')
        ax.text(label_seq.size + text_offset, 0.0, name, fontsize=10, fontname='Arial', va='center')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.xlim(0, max_seq_length*1.1)
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()

    return fig, axes


def plot_legend(out_file, num_classes=10, fig_width=4.5, fig_height_per_seq=0.3):
    box_width = 25
    text_offset = 10
    x_lim = 500

    figsize = (fig_width, (num_classes * 1) * fig_height_per_seq)
    fig, axes = plt.subplots(nrows=num_classes, ncols=1, sharex=True, figsize=figsize,
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    gesture_labels = list(np.arange(0, num_classes))
    gesture_descriptions = ['G1: Reaching for needle with right hand',
                            'G2: Positioning needle',
                            'G3: Pushing needle through tissue',
                            'G4: Transferring needle from left to right',
                            'G5: Moving to center with needle in grip',
                            'G6: Pulling suture with left hand',
                            'G8: Orienting needle',
                            'G9: Using right hand to help tighten suture',
                            'G10: Loosening more suture',
                            'G11: Dropping suture at end and moving to end points']

    for label, description, ax in zip(gesture_labels, gesture_descriptions, axes):
        plt.sca(ax)
        plt.axis('off')

        x = np.arange(0, box_width)
        y = 0 * np.ones(box_width)
        colors = label * np.ones(box_width)
        ax.scatter(x, y, c=colors, marker='|', s=200, lw=2, vmin=0, vmax=num_classes, cmap='tab10')
        ax.text(box_width + text_offset, 0.0, description, fontsize=10, fontname='Arial', va='center')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.xlim(0, x_lim*1.1)
        plt.ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def plot_confusion_matrix(out_file, model_dir, exp_group_name, eval_scheme, eval_freq, eval_type, model_no):
    eval_file = os.path.join(model_dir, "Eval", eval_scheme, "{}Hz".format(eval_freq), eval_type,
                             "{}.pth.tar".format(exp_group_name))
    eval_results = torch.load(eval_file)
    conf_mat = eval_results[model_no]['conf_mat']

    sums = np.sum(conf_mat, axis=1)
    conf_mat = conf_mat / sums[:,None]
    conf_mat = conf_mat * 100

    conf_mat_df = pd.DataFrame(conf_mat, index=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'],
                               columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'])

    plt.figure(figsize=(24, 18))
    sn.set(font_scale=2.2)
    sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 24}, cmap=plt.get_cmap('Blues'), cbar=False, square=True,
               fmt='.2f')
    plt.yticks(rotation=0)
    plt.ylabel("Actual gestures")
    plt.xlabel("Predicted gestures")
    plt.savefig(out_file)
