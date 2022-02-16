from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
import albumentations as A
from albumentations import BaseCompose
from albumentations.pytorch import transforms as AT


class DetectionMNIST(Dataset):
    """MNISTを加工した物体検出用データセット

    Args:
        Dataset (Dataset): torch.utils.data.Dataset
    """

    def __init__(self,
                 root: str,
                 threshold: float,
                 sigma: float,
                 transform: BaseCompose =None,
                 fashion: bool =False,
                 **kwargs) -> None:
        """

        Args:
            root (str): MNISTのルートディレクトリ
            threshold (float): 物体と背景の閾値
            sigma (float): キーポイントを作るガウス分布の径
            transform (BaseCompose, optional): 前処理とデータ拡張. Defaults to None.
            fashion (bool, optional): Fashion-MNISTを使うか. Defaults to False.
        """

        super().__init__()

        if fashion == True:
            self.mnist = FashionMNIST(root, **kwargs)
        else:
            self.mnist = MNIST(root, **kwargs)

        self.threshold = threshold
        self.sigma = sigma
        self.transform = transform

        self.grid_y, self.grid_x = torch.meshgrid(
            torch.arange(7), torch.arange(7), indexing='ij')    # ガウス分布に使用する座標

    def __len__(self) -> int:
        """データセットのサイズを取得する

        Returns:
            int: データセットのサイズ
        """
        return len(self.mnist)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """インデックスからデータを取得する

        Args:
            index (int): 取得するデータのインデックス

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            画像、バウンディングボックス、ラベル、キーポイントマップ、オフセットマップ、サイズマップ。
            画像のTensor形状は[1, 28, 28]
            バウンディングボックスのTensor形状は[M, 4]。Mはボックス個数だがM=1
            ラベルのTensor形状は[M]
            キーポイントマップのTensor形状は[10, 7, 7]
            オフセットマップのTensor形状は[2, 7, 7]
            サイズマップのTensor形状は[2, 7, 7]
        """
        image, label = self.mnist[index]

        image = np.array(image, dtype=np.float32) / 255.
        mask = np.where(image > self.threshold, True, False)    # 閾値を超えた画素がTrueのマスク

        y_indices, x_indices = np.where(mask)   # Trueになる画素の座標

        x_min = x_indices.min().astype(np.float32)
        y_min = y_indices.min().astype(np.float32)
        x_max = x_indices.max().astype(np.float32) + 1.     # ピクセルを含むように1加算
        y_max = y_indices.max().astype(np.float32) + 1.

        bbox = [x_min, y_min, x_max, y_max]

        bboxes = []     # 不定個のバウンディングボックスのリスト
        labels = []     # 不定個のラベルのリスト

        bboxes.append(bbox)     # 本来は物体の個数は不定だが、MNISTでは個数1で固定
        labels.append(label)

        # Albumentationsを使った前処理とデータ拡張
        if self.transform is not None:
            transformed = self.transform(
                image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        keypoint, offset, size = self._make_maps(bboxes, labels)    # マップの作成

        return image, bboxes, labels, keypoint, offset, size

    def _make_maps(self, bboxes: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """バウンディングボックスとラベルからマップを生成する

        Args:
            bboxes (Tensor): バウンディングボックス。Tensor形状は[M, 4]
            labels (Tensor): ラベル。Tensor形状は[M]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
            キーポイントマップ、オフセットマップ、サイズマップ
            キーポイントマップのTensor形状は[10, 7, 7]
            オフセットマップのTensor形状は[2, 7, 7]
            サイズマップのTensor形状は[2, 7, 7]
        """
        keypoint = torch.zeros([10, 7, 7], dtype=torch.float32)     # 1/4縮小のマップ
        offset = torch.zeros([2, 7, 7], dtype=torch.float32)        # xとyで2チャネル
        size = torch.zeros([2, 7, 7], dtype=torch.float32)          # 高さと幅で2チャネル

        # バウンディングボックスが複数あれば以下のループでボックス個数だけマップに加算する
        # 当然、MNISTでは以下のループは1回しか実行されないので注意
        for bbox, label in zip(bboxes, labels):
            x_center = (bbox[0] + bbox[2]) / 2.
            y_center = (bbox[1] + bbox[3]) / 2.
            W_index = int(x_center / 4.)    # x中心に対応するインデックス
            H_index = int(y_center / 4.)    # y中心に対応するインデックス

            # 中心位置が1のガウシアンをキーポイントマップに加算する
            keypoint[label] += torch.exp(-((self.grid_x - W_index) **
                                         2 + (self.grid_y - H_index)**2) / (2 * self.sigma**2))

            # オフセットの設定。キーポイントマップでのピクセル中心は元画像の2ピクセルずれた位置
            offset[0, H_index, W_index] += x_center - \
                4. * (W_index + 0.5)  # x offset
            offset[1, H_index, W_index] += y_center - \
                4. * (H_index + 0.5)  # y offset

            size[0, H_index, W_index] += bbox[2] - bbox[0]  # width
            size[1, H_index, W_index] += bbox[3] - bbox[1]  # height

        # 複数回の加算を考慮して、上限1下限0にスライスする
        keypoint = torch.clamp(keypoint, min=0., max=1.)    
        
        return keypoint, offset, size


if __name__ == '__main__':
    """テストコード
    """

    transform = A.Compose([
        AT.ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    sample_set = DetectionMNIST(root='/mnt/hdd/sika/Datasets', threshold=0.2, sigma=1.,
                                transform=transform, fashion=False,
                                train=True, download=False)
    
    print(f'dataset size: {len(sample_set)}')

    image, bboxes, labels, keypoint, offset, size = sample_set[0]

    print(f'image: {image.shape}  bbox: {bboxes.shape}  label: {labels.shape}')
    print(f'keypoint map: {keypoint.shape}  offset map: {offset.shape}  size map: {size.shape}')