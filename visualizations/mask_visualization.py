import math
from matplotlib import pyplot as plt
import torch
from torch import Tensor


def make_palette() -> Tensor:
    """マスク表示用のカラーパレットを作成する

    Returns:
        Tensor: パレット
    """
    tab10 = plt.get_cmap('tab10')

    cmap = [(0., 0., 0., 1.)]   # 先頭は背景
    for n in range(tab10.N):
        cmap.append(tab10(n))

    palette = torch.tensor(cmap)[:, :3]   # alphaは抜いてRGBのみ選択
    palette = (255 * palette).to(torch.long)

    return palette


def save_mask_image(images: Tensor, masks: Tensor, filename: str, nrows: int =10) -> None:
    """画像とマスクを画像ファイルに保存する

    Args:
        images (Tensor): 画像。Tensor形状は[N, C, H, W]
        masks (Tensor): マスク。Tensor形状は[N, H, W]
        filename (str): 保存するファイル名
        nrows (int, optional): 画像の表示行数. Defaults to 10.
    """
    plt.figure(figsize=(9, 9))
    palette = make_palette()

    ncols = math.ceil(len(images) / nrows)

    for i, (image, mask) in enumerate(zip(images, masks)):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.axis('off')

        ax.imshow(image.permute(1, 2, 0).numpy(), cmap='gray', vmin=0, vmax=1, extent=[0, 28, 28, 0])
        ax.imshow(palette[mask].numpy(), alpha=0.5, vmin=0, vmax=255, extent=[0, 28, 28, 0])

    plt.savefig(filename, facecolor='white')
    plt.close()