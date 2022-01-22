from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def get_peak_indices(keypoints: Tensor,
                     k: int =3,
                     threshold: float =0.5) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """キーポイントマップ中ピーク位置のインデックスを取得する

    Args:
        keypoints (Tensor): キーポイントマップ。Tensor形状は[N, C, H, W]
        k (int, optional): 検出するピーク位置の最大個数. Defaults to 3.
        threshold (float, optional): ピークの閾値. Defaults to 0.5.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
        第0軸インデックス、第1軸インデックス、第2軸インデックス、第3軸インデックス
    """
    dilation = F.max_pool2d(keypoints, 3, stride=1, padding=1)  # チャネル別の3x3のモルフォロジー膨張
    spatial_mask = (dilation == keypoints)  # 空間方向で3x3範囲の最大値に等しい位置がTrueのマスク

    channel_max = torch.max(keypoints, dim=1, keepdim=True)[0]     # ピクセル別のモルフォロジー膨張
    channel_mask = (channel_max == keypoints)  # チャネル方向で最大値に等しい位置がTrueのマスク

    max_mask = spatial_mask * channel_mask  # 3x3の空間範囲で最大かつ、チャネル方向でも最大がTrue
    peak = max_mask * keypoints # ピーク位置のみkeypointsの値、それ以外は0のピークマップ

    # それぞれの画像のk個のピーク位置
    values, indices = torch.topk(peak.flatten(start_dim=1), k=k, dim=1)

    # 画像ごとのk個のピークに対する第1軸、第2軸、第3軸のインデックス
    _, C_indices, H_indices, W_indices = np.unravel_index(indices.cpu().numpy(), peak.shape)
    C_indices = torch.tensor(C_indices)
    H_indices = torch.tensor(H_indices)
    W_indices = torch.tensor(W_indices)

    # それぞれフラット化する
    values = values.flatten()
    N_indices = torch.repeat_interleave(torch.arange(len(peak)), repeats=k) # 画像枚数×k
    C_indices = C_indices.flatten()
    H_indices = H_indices.flatten()
    W_indices = W_indices.flatten()

    # 閾値を超えたものだけ採用
    N_indices = N_indices[values > threshold]
    C_indices = C_indices[values > threshold]
    H_indices = H_indices[values > threshold]
    W_indices = W_indices[values > threshold]

    return N_indices, C_indices, H_indices, W_indices


def make_bboxes_list(keypoints: Tensor,
                     offsets: Tensor,
                     sizes: Tensor,
                     scale: float =4.,
                     k: int =3) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """ボックスとラベルのリストを作成する

    Args:
        keypoints (Tensor): キーポイントマップ。Tensor形状は[N, C, H, W]
        offsets (Tensor): オフセットマップ。Tensor形状は[N, 2, H, W]
        sizes (Tensor): サイズマップ。Tensor形状は[N, 2, H, W]
        scale (float, optional): マップから元画像サイズへの倍率. Defaults to 4.
        k (int, optional): 画像ごとの最大検出ボックス数. Defaults to 3.

    Returns:
        Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        ボックスのリスト、ラベルのリスト
    """
    # 画像枚数だけの空のリストを作成
    bboxes_list = [[] for _ in range(len(keypoints))]
    labels_list = [[] for _ in range(len(keypoints))]

    # ピーク位置のインデックス
    N_indices, C_indices, H_indices, W_indices = get_peak_indices(keypoints, k=k)
    
    for N_index, C_index, H_index, W_index in zip(N_indices, C_indices, H_indices, W_indices):
        # センター位置はscale個の画素の中心に対応するように、0.5を加算してから乗算
        x_center = scale * (W_index + 0.5).item()   
        y_center = scale * (H_index + 0.5).item()

        x_offset = offsets[N_index, 0, H_index, W_index].item()
        y_offset = offsets[N_index, 1, H_index, W_index].item()

        width = sizes[N_index, 0, H_index, W_index].item()
        height = sizes[N_index, 1, H_index, W_index].item()

        bbox = [x_center+x_offset-width/2., y_center+y_offset-height/2.,
                x_center+x_offset+width/2., y_center+y_offset+height/2.]

        bboxes_list[N_index.item()].append(bbox)
        labels_list[N_index.item()].append(C_index.item())

    for i in range(len(bboxes_list)):
        if bboxes_list[i] != []:    # 空じゃないものはTensor化
            bboxes_list[i] = torch.tensor(bboxes_list[i], dtype=torch.float32)
            labels_list[i] = torch.tensor(labels_list[i], dtype=torch.long)
        else:
            bboxes_list[i] = torch.empty([0, 4], dtype=torch.float32)
            labels_list[i] = torch.empty(0, dtype=torch.long)

    return bboxes_list, labels_list


if __name__ == '__main__':
    """テストコード
    """
    keypoints = torch.sigmoid(torch.rand([64, 10, 7, 7]))
    N_indices, C_indices, H_indices, W_indices = get_peak_indices(keypoints)

    offsets = torch.rand([64, 2, 7, 7]) - 0.5
    sizes = F.relu(torch.rand([64, 2, 7, 7]))

    bboxes_list, labales_list = make_bboxes_list(keypoints, offsets, sizes)
    print(f'bboxes_list: {len(bboxes_list)}  labels_list: {len(labales_list)}')
