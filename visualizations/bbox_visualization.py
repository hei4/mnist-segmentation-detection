from typing import Tuple
import math
from torch import Tensor
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


def save_bbox_image(images: Tensor,
                    predicted_bboxes_list: Tuple[Tensor, ...],
                    predicted_labels_list: Tuple[Tensor, ...],
                    target_bboxes_list: Tuple[Tensor, ...],
                    target_labels_list: Tuple[Tensor, ...],
                    filename: str,
                    nrows: int =10) -> None:
    """画像とバウンディングボックスを画像ファイルに保存する

    Args:
        images (Tensor): 画像。Tensor形状は[N, C, H, W]
        predicted_bboxes_list (Tuple[Tensor, ...]): 推論ボックスのリスト
        predicted_labels_list (Tuple[Tensor, ...]): 推論ラベルのリスト
        target_bboxes_list (Tuple[Tensor, ...]): 正解ボックスのリスト
        target_labels_list (Tuple[Tensor, ...]): 正解ラベルのリスト
        filename (str): 画像ファイル名
        nrows (int, optional): 画像の表示行数. Defaults to 10.
    """

    plt.figure(figsize=(9, 9))
    tab10 = plt.get_cmap('tab10')

    ncols = math.ceil(len(images) / nrows)

    for i, (image, target_bboxes, target_labels, predicted_bboxes, predicted_labels) in enumerate(zip(images, target_bboxes_list, target_labels_list, predicted_bboxes_list, predicted_labels_list)):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.axis('off')

        # 画像の描画
        ax.imshow(image.permute(1, 2, 0).numpy(), cmap='gray',
                  vmin=0, vmax=1, extent=[0, 28, 28, 0])

        # 正解ボックスの描画
        for bbox, label in zip(target_bboxes, target_labels):
            rect = mpatches.Rectangle([bbox[0].item(), bbox[1].item()],
                                      (bbox[2]-bbox[0]).item(),
                                      (bbox[3]-bbox[1]).item(),
                                      fill=False, edgecolor=tab10(label.item()),
                                      alpha=0.5, linewidth=1)
            ax.add_patch(rect)
        
        # 推論ボックスの描画
        for bbox, label in zip(predicted_bboxes, predicted_labels):
            rect = mpatches.Rectangle([bbox[0].item(), bbox[1].item()],
                                      (bbox[2]-bbox[0]).item(),
                                      (bbox[3]-bbox[1]).item(),
                                      fill=False, edgecolor=tab10(label.item()),
                                      linestyle='--', linewidth=1)
            ax.add_patch(rect)

    plt.savefig(filename, facecolor='white')
    plt.close()