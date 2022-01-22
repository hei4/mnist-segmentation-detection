import os
import argparse
from copy import deepcopy

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import transforms as AT

from dataloaders.detection_dataset import DetectionMNIST
from models.simple_pointnet import SimplePointNet
from criteria.pointnet_loss import PointNetLoss
from metrics.detection_metric import BboxScoring
from routines.detection_routine import run_detection


def main():
    parser = argparse.ArgumentParser(description='MNIST Object Detection')

    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dataroot', type=str, default='mnist',
                        help='path to dataset (default: mnist)')
    parser.add_argument('--pr_focal_loss', action='store_true',
                        help='use penalty reduced focal loss')
    
    args = parser.parse_args()

    # 前処理の定義
    transform = A.Compose([
        AT.ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    train_val_set = DetectionMNIST(root=args.dataroot, threshold=0.2, sigma=1., fashion=False,
                                   train=True, download=False, transform=transform)

    # 訓練データセット:検証データセット = 5:1 にランダム分割
    train_size = len(train_val_set) * 5 // 6
    val_size = len(train_val_set) - train_size
    train_set, val_set = random_split(train_val_set, [train_size, val_size])

    # 訓練用の前処理を設定するためにディープコピーで別オブジェクトに
    train_set.dataset = deepcopy(train_set.dataset)

    # 訓練セットにランダムアフィン変換のデータ拡張を設定
    train_set.dataset.transform = A.Compose([
        A.Affine(scale=(0.85, 1.15), translate_px=(-2, 2), rotate=(-10, 10),
                 shear=(-15, 15), always_apply=True),
        AT.ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # 画像と3つのマップは1つのTensorに結合し、ボックスとラベルはリストのままにする関数
    def collate_fn(batch):
        images, bboxes, labels, keypoints, offsets, sizes = zip(*batch)
        return torch.stack(images, dim=0), bboxes, labels, \
               torch.stack(keypoints, dim=0), torch.stack(offsets, dim=0), torch.stack(sizes, dim=0)

    # 訓練データローダーと検証データローダーを作成
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ネットワークの設定とGPUへの転送
    in_channels = 1
    num_classes = 10
    net = SimplePointNet(in_channels, num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 損失関数の設定
    criterion = PointNetLoss(use_pr_focal_loss=args.pr_focal_loss)

    # 評価関数の設定
    metric = BboxScoring(iou_threshold=0.75)

    # 最適化アルゴリズムの設定
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 保存先の作成
    os.makedirs('results/mnist_detection', exist_ok=True)

    for epoch in range(1, args.epochs+1):
        # 訓練実行
        run_detection('train', epoch, train_loader, device, net, criterion, metric, optimizer=optimizer)
        
        # 検証実行
        run_detection('val', epoch, val_loader, device, net, criterion, metric)

    # 学習したパラメータの保存
    net.to('cpu')
    file_name = f'results/mnist_detection/model_epoch{epoch}.pth'
    torch.save(net.state_dict(), file_name)


if __name__ == '__main__':
    main()
