"""CenterNetの公式実装を参考にしています https://github.com/xingyizhou/CenterNet
"""
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class PointNetLoss:
    """PointNet用の損失関数クラス

    キーポイント損失、サイズ損失、オフセット損失で構成される。
    キーポイント損失は、バイナリ交差損失とPointNet論文から実装したPenalty Reduced Focal Lossから選択できる
    """
    
    def __init__(self,
                 lambda_offset: float =1.,
                 lambda_size: float =0.1,
                 use_pr_focal_loss: bool =False,
                 alpha: float =2.,
                 beta: float=4.) -> None:
        """

        Args:
            lambda_offset (float, optional): オフセット損失の重み. Defaults to 1..
            lambda_size (float, optional): サイズ損失の重み. Defaults to 0.1.
            use_pr_focal_loss (bool, optional): Penalty Reduced Focal Lossを使うか. Defaults to False.
            alpha (float, optional): Penalty Reduced Focal Lossのハイパーパラメータ. Defaults to 2..
            beta (float, optional): Penalty Reduced Focal Lossのハイパーパラメータ. Defaults to 4..
        """
        
        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size

        if use_pr_focal_loss == False:
            self.keypoint_loss = nn.BCELoss(reduction='sum')
        else:
            self.keypoint_loss = self._penalty_reduced_focal_loss

        self.alpha = alpha
        self.beta = beta
    
    def __call__(self,
                predicted_keypoints: Tensor,
                predicted_offsets: Tensor,
                predicted_sizes: Tensor,
                target_keypoints: Tensor,
                target_offsets: Tensor,
                target_sizes: Tensor) -> Tensor:
        """損失値を計算する

        Args:
            predicted_keypoints (Tensor): 推定したキーポイントマップ。Tensor形状は[N, C, H, W]
            predicted_offsets (Tensor): 推定したオフセットマップ。Tensor形状は[N, 2, H, W]
            predicted_sizes (Tensor): 推定したサイズマップ。Tensor形状は[N, 2, H, W]
            target_keypoints (Tensor): ターゲットのキーポイントマップ。Tensor形状は[N, C, H, W]
            target_offsets (Tensor): ターゲットのオフセットマップ。Tensor形状は[N, 2, H, W]
            target_sizes (Tensor): ターゲットのサイズマップ。Tensor形状は[N, 2, H, W]

        Returns:
            Tensor: 損失値。Tensor形状はスカラー
        """

        # キーポイント損失。初期化メソッドで実体を設定している
        loss_keypoint = self.keypoint_loss(predicted_keypoints, target_keypoints)

        # オフセット損失とサイズ損失のためにピークのみTrueのマスクを作成する
        peak_mask = target_keypoints == 1.  # 1の位置がガウシアン中心のピーク
        peak_mask = torch.sum(peak_mask, dim=1).to(torch.bool)  # チャネル方向の論理和
        peak_mask = torch.repeat_interleave(peak_mask.unsqueeze(dim=1),
                                            2,
                                            dim=1)  # L1損失のためにチャネル方向に×2

        loss_offset = F.l1_loss(predicted_offsets * peak_mask,
                                target_offsets * peak_mask, reduction='sum')

        loss_size = F.l1_loss(predicted_sizes * peak_mask,
                              target_sizes * peak_mask, reduction='sum')

        # ピーク数で除算して平均にする
        num_peak = torch.sum(peak_mask)
        if num_peak > 0:
            loss_keypoint /= num_peak
            loss_offset /= num_peak
            loss_size /= num_peak

        return loss_keypoint + self.lambda_offset * loss_offset + self.lambda_size * loss_size

    def _penalty_reduced_focal_loss(self,
                                    predicted_keypoints: Tensor,
                                    target_keypoints: Tensor) -> Tensor:
        """PointNet論文に記載のPenalty Reduced Focal Loss

        Args:
            predicted_keypoints (Tensor): 推定したキーポイントマップ。Tensor形状は[N, C, H, W]
            target_keypoints (Tensor): ターゲットのキーポイントマップ。Tensor形状は[N, C, H, W]

        Returns:
            Tensor: 損失値。Tensor形状はスカラー
        """

        # ピークのみTrueのマスクを作成する
        peak_mask = target_keypoints == 1.  # 1の位置がガウシアン中心のピーク

        # ピーク位置の損失
        positive = -((1. - predicted_keypoints) ** self.alpha) * \
            torch.log(predicted_keypoints + 1e-6)
        positive = torch.sum(positive[peak_mask])   # マスクがTrueの箇所のみ加算

        reverse_mask = torch.logical_not(peak_mask)     # マスク反転

        # ピーク位置以外の損失
        negative = -((1. - target_keypoints) ** self.beta) * \
            (predicted_keypoints ** self.alpha) * \
            torch.log(1. - predicted_keypoints + 1e-6)
        negative = torch.sum(negative[reverse_mask])    # 反転マスクがTrueの箇所のみ加算
        
        return positive + negative


if __name__ == '__main__':
    """テストコード
    """
    N = 64
    C = 10
    H = 128
    W = 128

    predicted_keypoints = torch.sigmoid(torch.rand(N, C, H, W))
    predicted_offsets = 2. * (torch.rand(N, 2, H, W) - 0.5)
    predicted_sizes = F.relu(torch.rand(N, 2, H, W))

    target_keypoints = torch.sigmoid(torch.rand(N, C, H, W))
    target_offsets = 2. * (torch.rand(N, 2, H, W) - 0.5)
    target_sizes = F.relu(torch.rand(N, 2, H, W))

    criterion = PointNetLoss()

    loss = criterion(predicted_keypoints,
                     predicted_offsets,
                     predicted_sizes,
                     target_keypoints,
                     target_offsets,
                     target_sizes)
    
    print(f'loss: {loss.item()}')
