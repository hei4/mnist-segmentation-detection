
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module

class SimpleUNet(Module):
    """シンプルなUNet

    Args:
        Module (Module): torch.nn.Module
    """
    def __init__(self, in_channels: int =1, num_classes: int =11) -> None:
        """初期化メソッド

        Args:
            in_channels (int, optional): 入力チャネル数. Defaults to 1.
            num_classes (int, optional): クラス数. Defaults to 11.
        """
        super().__init__()

        # エンコーダー部
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),     # [N, 64, 28, 28]
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(2),    # [N, 64, 14, 14]

                nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),     # [N, 128, 14, 14]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 14, 14]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])

        # ボトルネック部
        self.bottle_neck = nn.Sequential(
            nn.MaxPool2d(2),    # [N, 128, 7, 7]

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # [N, 128, 14, 14]
        )

        # デコーダー部
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128+256, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 14, 14]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 14, 14]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # [N, 128, 28, 28]
            ),

            nn.Sequential(
                nn.Conv2d(64+128, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 28, 28]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, num_classes, 1, stride=1, padding=0)     # [N, 11, 28, 28]
            )
        ])

    def forward(self, x: Tensor) -> Tensor:
        """順伝播を行う

        Args:
            x (Tensor): 画像。Tensor形状は[N, in_channels, H, W]

        Returns:
            Tensor: ロジット。Tensor形状は[N, num_classes, H, W]
        """
        h_list = []     # 特徴マップを格納するリスト
        for block in self.encoder:
            x = block(x)
            h_list.append(x)

        x = self.bottle_neck(x)

        for h, block in zip(h_list[::-1], self.decoder):    # 特徴マップの逆順とデコーダーをzip
            x = block(torch.cat([h, x], dim=1))

        return x


if __name__ == '__main__':
    """テストコード
    """
    N = 64
    in_channel = 1
    num_classes = 11
    H = 128
    W = 128
    
    images = torch.rand(N, in_channel, H, W)
    print(f'images: {images.shape}')

    net = SimpleUNet(in_channel, num_classes)

    logits = net(images)
    print(f'logits: {logits.shape}')


