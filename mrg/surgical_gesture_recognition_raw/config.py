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
parser.add_argument('--sur_task_name', type=str, default='su',
                    help="task_name.")
parser.add_argument('--seq_limit', type=int, default=300,
                    help="upper bound of  seq len.")
parser.add_argument('--enc_in', type=int, default=64, help='encoder input size for kinetics data')
parser.add_argument('--model_type', type=str, default='hybrid', help='model type: pure kinetics,pure vision,hybrid model')
# newly add
parser.add_argument('--model', type=str,  default='TimesNet', help='model name, options: [Autoformer, Informer,Transformer, TimesNet]')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--task_name', type=str,  default='encoder', help='t')
parser.add_argument('--pred_len', type=int, default=0, help='default len should be zero for encoder task')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--d_model', type=int, default=72, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--embed', type=str, default='none', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.01, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--c_out', type=int, default=12, help='当做encoder使用时候是不是需要等于啥，存疑')
parser.add_argument('--moving_avg', type=int, default=50, help='window size of moving average')
# for time net
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
# useless
parser.add_argument('--d_layers', type=int, default=1, help='useless')
parser.add_argument('--dec_in', type=int, default=7, help='useless')
parser.add_argument('--freq', type=str, default='h', help='useless')
parser.add_argument('--output_attention', action='store_true', help='useless')
parser.add_argument('--distil', action='store_false', help='useless',default=True)
# for model build
parser.add_argument('--fusion_type', type=str,  default='3')
# parser.add_argument('--c_out', type=int, default=12, help='output size') # MICN is affected by this # cout 存疑
# --seq_len 20 --label_len 20 --pred_len 0 --e_layers 1 --d_layers 1 --factor 2 --enc_in 4 --dec_in 2 --c_out 2 --d_model 16 --d_ff 16 --des 'Exp' --itr 1 --top_k 2 --model Informer
args = parser.parse_args()
if args.sur_task_name == 'su':
    args.num_classes = 10
    gesture_class_num = 10
elif args.sur_task_name == 'kt':
    args.num_classes = 6
    gesture_class_num = 6
elif args.sur_task_name == 'np':
    args.num_classes = 8
    gesture_class_num = 8

setting = '{}_{}_dm{}_nh{}_el{}_df{}_fc{}_ma{}'.format(
            args.sur_task_name,
            args.split,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.factor,
            args.moving_avg,
            )
