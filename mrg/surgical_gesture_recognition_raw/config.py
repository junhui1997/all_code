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
parser.add_argument('--seq_limit', type=int, default=300,
                    help="upper bound of  seq len.")
parser.add_argument('--enc_in', type=int, default=14, help='encoder input size for kinetics data')
parser.add_argument('--model_type', type=str, default='hybrid', help='model type: pure kinetics,pure vision,hybrid model')
# newly add
parser.add_argument('--model', type=str,  default='Informer', help='model name, options: [Autoformer, Transformer, TimesNet]')
parser.add_argument('--task_name', type=str, required=True, default='encoder', help='t')
parser.add_argument('--pred_len', type=int, default=0, help='default len should be zero for encoder task')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--d_model', type=int, default=72, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
# parser.add_argument('--c_out', type=int, default=12, help='output size') # MICN is affected by this # cout 存疑
# --seq_len 20 --label_len 20 --pred_len 0 --e_layers 1 --d_layers 1 --factor 2 --enc_in 4 --dec_in 2 --c_out 2 --d_model 16 --d_ff 16 --des 'Exp' --itr 1 --top_k 2 --model Informer
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