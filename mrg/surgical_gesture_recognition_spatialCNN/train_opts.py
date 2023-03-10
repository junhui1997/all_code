import argparse


num_cls_Kinetics = 400


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="Train model for video-based surgical gesture recognition.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, required=True, help="Name (description) of the experiment to run.")
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Suturing',
                    help="JIGSAWS task to evaluate.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOUO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO).")
parser.add_argument('--split', type=int, required=True, help="Cross-validation fold (data split) to evaluate.")
parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'], help="Used input modality.")

# Data
parser.add_argument('--data_path', type=str, default="../../Suturing/frames/",
                    help="Path to data folder, which contains the extracted images for each video. "
                         "One subfolder per video.")
parser.add_argument('--transcriptions_dir', type=str, default="../../Suturing/transcriptions/",
                    help="Path to folder containing the transcription files (gesture annotations). One file per video.")
# kinematics data
parser.add_argument('--kinematics_dir', type=str, default="../../Suturing/kinematics/",
                    help="Path to folder containing the kinematics data. One file per video.")
parser.add_argument('--vision_dir', type=str, default="?",
                    help="Path to folder containing the vision features. One file per video.")

parser.add_argument('--video_lists_dir', type=str, default="./Splits/{}/",
                    help="Path to directory containing information about each video in the form of video list files. "
                         "One subfolder per evaluation scheme, one file per evaluation fold.")
parser.add_argument('--video_sampling_step', type=int, default=3,
                    help="Describes how the available video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
parser.add_argument('--snippet_sampling_step', type=int, default=3,
                    help="snippet_sampling_step")                         
parser.add_argument('--do_horizontal_flip', type='bool', default=False,
                    help="Whether data augmentation should include a random horizontal flip.")
parser.add_argument('--corner_cropping', type='bool', default=True,
                    help="Whether data augmentation should include corner cropping.")
parser.add_argument('--data_preloading', type='bool', default=False,
                    help="Whether all image data should be loaded to RAM before starting network training.")

# Model
parser.add_argument('--snippet_length', type=int, default=1, help="Number of frames constituting one video snippet.")
parser.add_argument('--input_size', type=int, default=224, help="Target size (width/ height) of each frame.")
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability applied at final dropout layer.")

# Training
parser.add_argument('--resume_exp', type=str, default=None,
                    help="Path to results of former experiment that shall be resumed (untested).")
parser.add_argument('--bootstrap_from_2D', type='bool', default=False,
                    help="Whether model weights are to be bootstrapped from a previously trained 2D model.")
parser.add_argument('--pretrain_path', type=str, default=None,
                    help="Path to pretrained model weights. If <bootstrap_from_2D> is true, this should be the path to "
                         "the results folder of a previously run experiment.")
parser.add_argument('--pretrained_2D_model_no', type=int, default=249,
                    help="If <bootstrap_from_2D> is true, the models trained for (<pretrained_2D_model_no> + 1) epochs "
                         "will be used for weight initialization.")
parser.add_argument('-j', '--workers', type=int, default=4, help="Number of threads used for data loading.")
parser.add_argument('--steps_per_epoch', type=int, default=3000,
                    help="Minimum number of gesture snippets to process during one epoch of training.")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
parser.add_argument('-b', '--batch-size', type=int, default=64, help="Batch size.")
parser.add_argument('--lr', type=float, default=0.0025, help="Learning rate.")
parser.add_argument('--use_scheduler', type=bool, default=True, help="Whether to use the learning rate scheduler.")
parser.add_argument('--scheduler_step', type=int, default=50, help="learning rate scheduler step size.")
parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="learning rate scheduler gamma.")
parser.add_argument('--loss_weighting', type=bool, default=True,
                    help="Whether to apply weights to loss calculation so that errors in more current predictions "
                         "weigh more heavily.")
parser.add_argument('--eval_freq', '-ef', type=int, default=1, help="Validate model every <eval_freq> epochs.")
parser.add_argument('--save_freq', '-sf', type=int, default=5, help="Save checkpoint every <save_freq> epochs.")
parser.add_argument('--out', type=str, required=True,
                    help="Path to output folder, where all models and results will be stored.")
