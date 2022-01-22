import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from albumentations.pytorch import transforms as AT

from dataloaders.segmentation_dataset import SegmentationMNIST
from models.simple_unet import SimpleUNet
from metrics.segmentation_metric import MaskScoring
from routines.segmentation_routine import run_segmentation 


def main():
    parser = argparse.ArgumentParser(description='PyTorch Miminal Semantic Segmentation (inference)')

    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--dataroot', type=str, default='/mnt/hdd/sika/Datasets',# default='data',
                        help='path to dataset')
    parser.add_argument('--param', type=str, default='results/mnist_segmentation/model_epoch20.pth',
                        help='path to parameter')

    args = parser.parse_args()

    transform = AT.ToTensorV2()     # 前処理の設定
    test_set = SegmentationMNIST(root=args.dataroot, threshold=0.2, fashion=False,
                                 train=False, download=False, transform=transform)

    # テストデータローダーを作成
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ネットワークの設定
    in_channels = 1
    num_foreground = 10
    num_classes = num_foreground + 1
    net = SimpleUNet(in_channels, num_classes)
    net.load_state_dict(torch.load(args.param))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # 評価関数の設定
    metric = MaskScoring(num_classes=num_classes)

    # テストローダーについて推論実行
    run_segmentation('test', None, test_loader, device, net, criterion, metric)

if __name__ == '__main__':
    main()