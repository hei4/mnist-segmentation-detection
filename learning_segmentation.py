import os
import argparse
from copy import deepcopy

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import transforms as AT

from dataloaders.segmentation_dataset import SegmentationMNIST
from models.simple_unet import SimpleUNet
from metrics.segmentation_metric import MaskScoring
from routines.segmentation_routine import run_segmentation


def main():
    parser = argparse.ArgumentParser(description='PyTorch Miminal Semantic Segmentation')

    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataroot', type=str, default='/mnt/hdd/sika/Datasets',# default='data',
                        help='path to dataset')
    args = parser.parse_args()

    transform = AT.ToTensorV2()     # 前処理の設定
    train_val_set = SegmentationMNIST(root=args.dataroot, threshold=0.2, fashion=False,
                                      train=True, download=False, transform=transform)

    # 訓練データセット:検証データセット = 5:1 にランダム分割
    train_size = len(train_val_set) * 5 // 6
    val_size = len(train_val_set) - train_size
    train_set, val_set = random_split(train_val_set, [train_size, val_size])

    # 訓練用の前処理を設定するためにディープコピーで別オブジェクトに
    train_set.dataset = deepcopy(train_set.dataset)
    
    # 訓練セットにランダムアフィン変換のデータ拡張を設定
    train_transform = A.Compose([
        A.Affine(scale=(0.85, 1.15), translate_px=(-2, 2), rotate=(-10, 10), shear=(-15, 15), always_apply=True),
        AT.ToTensorV2()
    ])
    train_set.dataset.transform = train_transform

    # 訓練データローダーと検証データローダーを作成
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ネットワークの設定とGPUへの転送
    in_channels = 1
    num_foreground = 10
    num_classes = num_foreground + 1    # 背景+物体クラスで11
    net = SimpleUNet(in_channels, num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # 評価関数の設定
    metric = MaskScoring(num_classes=num_classes)

    # 最適化アルゴリズムの設定
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 保存先の作成
    os.makedirs('results/mnist_segmentation', exist_ok=True)

    for epoch in range(1, args.epochs+1):
        # 訓練実行
        run_segmentation('train', epoch, train_loader, device, net, criterion, metric, optimizer=optimizer)
        
        # 検証実行
        run_segmentation('val', epoch, val_loader, device, net, criterion, metric)

    net.to('cpu')
    torch.save(net.state_dict(), f'results/mnist_segmentation/model_epoch{epoch}.pth')


if __name__ == '__main__':
    main()