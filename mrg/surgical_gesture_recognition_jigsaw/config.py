from __future__ import division
from __future__ import print_function
import argparse
import json
import pdb
import math

tcn_run_num = None

tcn_params = None
rl_params = None
sample_rate = None
gesture_class_num = None
kinematics_dir = None
transcriptions_dir = None

raw_feature_dir = None
result_dir = None
tcn_log_dir = None
tcn_model_dir = None
tcn_feature_dir = None
trpo_model_dir = None
graph_dir = None

split_info_dir = None

all_params = json.load(open('config.json'))

dataset_name = all_params['dataset_name']

locals().update(all_params['experiment_setup'])
locals().update(all_params[dataset_name])


tcn_params['model_params']['encoder_params']['kernel_size'] //= sample_rate
tcn_params['model_params']['decoder_params']['kernel_size'] //= sample_rate

tcn_params['model_params']['mid_lstm_params'] = None

temp = []
for k in rl_params['k_steps']:
    temp.append(math.ceil(k / sample_rate))
rl_params['k_steps'] = temp

temp = []
for k in rl_params['glimpse']:
    temp.append(math.ceil(k / sample_rate))
rl_params['glimpse'] = temp

parser = argparse.ArgumentParser(description="Train model for video-based surgical gesture recognition.")

parser.add_argument('--split', type=int, required=True,
                    choices=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="Cross-validation fold (data split) to evaluate.")
parser.add_argument('--save_name', type=str, required=True,
                    help="save name.")
parser.add_argument('--task_name', type=str, required=True,
                    help="task_name.")
parser.add_argument('--seq_limit', type=int, required=True,
                    help="upper bound of  seq len.")
parser.add_argument('--enc_in', type=int, default=14, help='encoder input size for kinetics data')
args = parser.parse_args()
if args.task_name == 'su':
    args.num_classes = 10
    gesture_class_num = 10
elif args.task_name == 'kt':
    args.num_classes = 6
    gesture_class_num = 6
elif args.task_name == 'np':
    args.num_classes = 8
    gesture_class_num = 8