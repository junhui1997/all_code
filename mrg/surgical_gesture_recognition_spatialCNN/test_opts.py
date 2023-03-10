import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="Test model for video-based surgical gesture recognition.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, required=True,
                    help="Name of the experiment to evaluate (including auto-generated timestamp).")
parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Needle_Passing',
                    help="JIGSAWS task to evaluate.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOUO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO).")
parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'], help="Used input modality.")

# Data
parser.add_argument('--data_path', type=str, default="../../Needle_Passing/frames/",
                    help="Path to data folder, which contains the extracted images for each video. "
                         "One subfolder per video.")
parser.add_argument('--transcriptions_dir', type=str, default="../../Needle_Passing/transcriptions/",
                    help="Path to folder containing the transcription files (gesture annotations). One file per video.")
# kinematics data
parser.add_argument('--kinematics_dir', type=str, default="../../Needle_Passing/kinematics/",
                    help="Path to folder containing the kinematics data. One file per video.")

parser.add_argument('--video_lists_dir', type=str, default="./Splits/{}/",
                    help="Path to directory containing information about each video in the form of video list files. "
                         "One subfolder per evaluation scheme, one file per evaluation fold.")
parser.add_argument('--video_sampling_step', type=int, default=3, choices=[3, 6, 15],
                    help="Specifies at which temporal resolution each video will be evaluated. "
                         "More specifically, we will sample one video snippet every <video_sampling_step>th frame.")
parser.add_argument('--snippet_sampling_step', type=int, default=3,
                    help="Specifies at which temporal resolution the video snippets will be created (by taking every "
                         "<snippet_sampling_step>th video frame). ")

# Model
parser.add_argument('--use_resnet_shortcut_type_B', type='bool', default=False,
                    help="Whether to use shortcut connections of type B.")
parser.add_argument('--return_feature', type='bool', default=False,
                    help="Whether to return the fc hidden feature (128-dim)")
parser.add_argument('--snippet_length', type=int, default=1, help="Number of frames constituting one video snippet.")
parser.add_argument('--input_size', type=int, default=224, help="Target size (width/ height) of each frame.")

# Testing
parser.add_argument('--model_dir', type=str, default="./result/",
                    help="Path to the folder where the models for the relevant experiment(s) are stored. "
                         "Usually identical to <out> as specified during training.")
parser.add_argument('--model_no', type=int, nargs="+", default=[-1],
                    help="Defines the models to evaluate by specifying their model numbers. Will perform one evaluation"
                         " run per model number. The number of a model corresponds to the number of epochs for which "
                         "the model has been trained - 1.")
parser.add_argument('--sliding_window', type='bool', default=False, help="Whether to accumulate predictions over time.")
parser.add_argument('--look_ahead', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    help="If <sliding_window> is true, predictions for future snippets are considered only if "
                         "they are at most <look_ahead> steps away.")
parser.add_argument('-j', '--workers', type=int, default=4, help="Number of threads used for data loading.")
parser.add_argument('-b', '--batch-size', type=int, default=128, help="Batch size.")
