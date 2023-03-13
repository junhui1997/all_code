import argparse
import torch
import torch.nn as nn
from my_dataset import RawFeatureDataset
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Train model for video-based surgical gesture recognition.")

parser.add_argument('--model', type=str, required=True, help="the model to extract feature")
parser.add_argument('--task_name', type=str, required=True, help="to select which task")
args = parser.parse_args()

def main():
    train_dataset = RawFeatureDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()
