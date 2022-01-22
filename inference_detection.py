import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import transforms as AT

from dataloaders.detection_dataset import DetectionMNIST
from models.simple_pointnet import SimplePointNet
from criteria.pointnet_loss import PointNetLoss
from metrics.detection_metric import BboxScoring
from learning_detection import run_detection


def main():
    parser = argparse.ArgumentParser(description='MNIST Object Detection (inference)')

    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--dataroot', type=str, default='mnist',
                        help='path to dataset')
    parser.add_argument('--pr_focal_loss', action='store_true',
                        help='use penalty reduced focal loss')
    parser.add_argument('--param', type=str, default='results/mnist_detection/model_epoch20.pth',
                        help='path to parameter')

    args = parser.parse_args()

    # 前処理の定義
    transform = A.Compose([
        AT.ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    test_set = DetectionMNIST(root=args.dataroot, threshold=0.2, sigma=1., fashion=False,
                              train=False, download=False, transform=transform)

    def collate_fn(batch):
        images, bboxes, labels, keypoints, offsets, sizes = zip(*batch)
        return torch.stack(images, dim=0), bboxes, labels, \
               torch.stack(keypoints, dim=0), torch.stack(offsets, dim=0), torch.stack(sizes, dim=0)

    # テストデータローダーを作成
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ネットワークの設定
    in_channels = 1
    num_classes = 10
    net = SimplePointNet(in_channels, num_classes)
    net.load_state_dict(torch.load(args.param))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 損失関数の設定
    criterion = PointNetLoss(use_pr_focal_loss=args.pr_focal_loss)

    # 評価関数の設定
    metric = BboxScoring(iou_threshold=0.75)

    # テストローダーについて推論実行
    run_detection('test', None, test_loader, device, net, criterion, metric)


if __name__ == '__main__':
    main()