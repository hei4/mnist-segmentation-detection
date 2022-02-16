from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
import albumentations as A
from albumentations import BaseCompose
from albumentations.pytorch import transforms as AT


class SegmentationMNIST(Dataset):
    """MNISTを加工したセマンティックセグメンテーション用データセット

    Args:
        Dataset (Dataset): torch.utils.data.Dataset
    """
    def __init__(self,
                 root: str,
                 threshold: float,
                 transform: BaseCompose =None,
                 fashion: bool =False,
                 **kwargs) -> None:
        """

        Args:
            root (str): MNISTのルートディレクトリ
            threshold (float): 物体と背景の閾値
            transform (BaseCompose): 前処理とデータ拡張. Defaults to None.
            fashion (bool, optional): Fashion-MNISTを使うか. Defaults to False.
        """

        super().__init__()

        if fashion == True:
            self.mnist = FashionMNIST(root, **kwargs)
        else:
            self.mnist = MNIST(root, **kwargs)

        self.threshold = threshold
        self.transform = transform
    
    def __len__(self) -> int:
        """データセットのサイズを取得する

        Returns:
            int: データセットのサイズ
        """
        return len(self.mnist)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """インデックスからデータを取得する

        Args:
            index (int): 取得するデータのインデックス

        Returns:
            Tuple[Tensor, Tensor]:
            画像、マスク。
            画像のTensor形状は[1, 28, 28]
            マスクのTensor形状は[28, 28]
        """
        image, label = self.mnist[index]

        image = np.array(image, dtype=np.float32) / 255.
        label += 1  # 背景を0にするのでインクリメント
        mask = np.where(image > self.threshold, label, 0)   # 閾値を超えた画素にラベルの値

        # Albumentationsを使った前処理とデータ拡張
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].to(torch.int64)

        return image, mask


if __name__ == '__main__':
    """テストコード
    """
    transform = AT.ToTensorV2()

    sample_set = SegmentationMNIST(root='/mnt/hdd/sika/Datasets', threshold=0.2,
                                   transform=transform, fashion=False,
                                   train=True, download=True)
    
    print(f'dataset size: {len(sample_set)}')

    image, mask = sample_set[0]

    print(f'image: {image.shape}  mask: {mask.shape}')