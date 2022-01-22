import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as F


class PointNetLoss(Module):
    """PointNet用の損失関数クラス

    キーポイント損失、サイズ損失、オフセット損失で構成される。
    キーポイント損失は、バイナリ交差損失とPointNet論文から実装した損失から選択できる

    Args:
        Module (Module): torch.nn.Module
    """
    def __init__(self,
                 lambda_offset: float =1.,
                 lambda_size: float =0.1,
                 use_pr_focal_loss: bool =False,
                 alpha: float =2.,
                 beta: float=4.) -> None:
        """初期化メソッド

        Args:
            lambda_offset (float, optional): オフセット損失の重み. Defaults to 1..
            lambda_size (float, optional): サイズ損失の重み. Defaults to 0.1.
            use_pr_focal_loss (bool, optional): PointNet論文から実装した損失. Defaults to False.
            alpha (float, optional): Penalty Reduced Focal Lossのハイパーパラメータ. Defaults to 2..
            beta (float, optional): Penalty Reduced Focal Lossのハイパーパラメータ. Defaults to 4..
        """
        
        super().__init__()
        
        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size

        if use_pr_focal_loss == False:
            self.keypoint_loss = nn.BCELoss(reduction='sum')
        else:
            self.keypoint_loss = self._penalty_reduced_focal_loss

        self.alpha = alpha
        self.beta = beta
    
    def forward(self,
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

        loss_keypoint = self.keypoint_loss(predicted_keypoints, target_keypoints)

        peak_mask = target_keypoints == 1.
        peak_mask = torch.sum(peak_mask, dim=1).to(torch.bool)
        peak_mask = torch.repeat_interleave(peak_mask.unsqueeze(dim=1), 2, dim=1)

        loss_offset = F.l1_loss(predicted_offsets * peak_mask,
                                target_offsets * peak_mask, reduction='sum')

        loss_size = F.l1_loss(predicted_sizes * peak_mask,
                              target_sizes * peak_mask, reduction='sum')

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

        # ターゲットが1になっている箇所のみTrueのマスク
        peak_mask = torch.where(1.-target_keypoints < 1e-6, True, False)

        positive = -((1. - predicted_keypoints) ** self.alpha) * \
            torch.log(predicted_keypoints + 1e-6)
        
        positive = torch.sum(positive[peak_mask])   # マスクTrueの箇所のみ加算

        reverse_mask = torch.logical_not(peak_mask)     # マスク反転

        negative = -((1. - target_keypoints) ** self.beta) * \
            (predicted_keypoints ** self.alpha) * \
            torch.log(1. - predicted_keypoints + 1e-6)
        
        negative = torch.sum(negative[reverse_mask])    # 反転マスクTrueの箇所のみ加算
        
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
