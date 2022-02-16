from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module


class SimplePointNet(Module):
    """シンプルなPointNet

    Args:
        Module (Module): torch.nn.Module
    """

    def __init__(self, in_channels: int =1, num_classes: int =10) -> None:
        """

        Args:
            in_channels (int, optional): 入力チャネル数. Defaults to 1.
            num_classes (int, optional): クラス数. Defaults to 10.
        """
        super().__init__()

        self.num_classes = num_classes

        # キーポイント、オフセット、サイズで共通して使うバックボーン
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),     # [N, 64, 28, 28]
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),     # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),     # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # キーポイントマップは値が0～1なので最後はシグモイド
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, self.num_classes, 1, stride=1, padding=0),  # [N, 10, 7, 7]
            nn.Sigmoid()
        )

        # オフセットマップは値が正負をとるので最後は恒等写像（Conv2dのまま）
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 2, 1, stride=1, padding=0),  # [N, 2, 7, 7]
        )
        
        # サイズマップは値が0～なので最後はReLU
        self.size_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 2, 1, stride=1, padding=0),   # [N, 2, 7, 7]
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """順伝播を行う

        Args:
            x (Tensor): 入力。Tensor形状は[N, in_channels, H, W]

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            キーポイントマップ、オフセットマップ、サイズマップ。
            Tensor形状は[N, num_classes, H/4, W/4]、[N, 2, H/4, W/4]、[N, 2, H/4, W/4]
        """
        h = self.backbone(x)

        return self.keypoint_head(h), self.offset_head(h), self.size_head(h)


if __name__ == '__main__':
    """テストコード
    """
    N = 64
    in_channel = 1
    num_classes = 10
    H = 128
    W = 128
    
    images = torch.rand(N, in_channel, H, W)
    print(f'images: {images.shape}')

    net = SimplePointNet(in_channel, num_classes)

    keypoints, offsets, sizes = net(images)
    print(f'keypoints: {keypoints.shape}  offsets: {offsets.shape}  sizes: {sizes.shape}')



