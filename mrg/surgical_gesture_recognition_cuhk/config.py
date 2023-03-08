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
                    choices=[1, 2, 3, 4],
                    help="Cross-validation fold (data split) to evaluate.")
parser.add_argument('--save_name', type=str, required=True,
                    help="save name.")
args = parser.parse_args()