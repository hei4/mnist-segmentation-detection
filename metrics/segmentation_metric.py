from typing import Tuple
import torch
from torch import Tensor
from torch.nn import functional as F


class MaskScoring:
    """セマンティックセグメンテーションのTP・FP・FNを評価するクラス
    """
    def __init__(self, num_classes: int) -> None:
        """

        Args:
            num_classes (int): クラス数
        """
        self.num_classes = num_classes
    
    def __call__(self,
                 predicted_masks: Tensor,
                 target_masks: Tensor) -> Tuple[float, float, float]:
        """ミニバッチの真陽性、偽陽性、偽陰性を計算する

        Args:
            predicted_masks (Tensor): 推論マスク。Tensor形状は[N, H, W]
            target_masks (Tensor): 正解マスク。Tensor形状は[N, H, W]

        Returns:
            Tuple[float, float, float]: ミニバッチの真陽性、偽陽性、偽陰性
        """
        # One-hotエンコーディング。クラス数の軸は末尾になる
        target_masks = F.one_hot(target_masks, self.num_classes)   # [N, H, W, C]
        predicted_masks = F.one_hot(predicted_masks, self.num_classes)
        
        # 前景クラスでチャネルを選択する
        target_masks = target_masks[:, :, :, 1:]  # [N, H, W, C-1]
        predicted_masks = predicted_masks[:, :, :, 1:]

        # 論理演算を行う
        minibatch_tp = torch.logical_and(target_masks, predicted_masks).sum()
        minibatch_fp = torch.logical_and(torch.logical_not(target_masks), predicted_masks).sum()
        minibatch_fn = torch.logical_and(target_masks, torch.logical_not(predicted_masks)).sum()
        
        return minibatch_tp.item(), minibatch_fp.item(), minibatch_fn.item()


if __name__ == '__main__':
    """テストコード
    """
    N = 64
    H = 128
    W = 128
    num_classes = 10
    
    predicted_masks = torch.randint(low=0, high=num_classes, size=[N, H, W])
    target_masks = torch.randint(low=0, high=num_classes, size=[N, H, W])

    metrics = MaskScoring(num_classes)
    minibatch_tp, minibatch_fp, minibatch_fn = metrics(predicted_masks, target_masks)

    print(f'TP: {minibatch_tp}  FP: {minibatch_fp}  FN: {minibatch_fn}')