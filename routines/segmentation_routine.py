import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.simple_unet import SimpleUNet
from metrics.segmentation_metric import MaskScoring
from visualizations.mask_visualization import save_mask_image


def run_segmentation(mode: str,
                     epoch: int,
                     loader: DataLoader,
                     device: torch.device,
                     net: SimpleUNet,
                     criterion: nn.CrossEntropyLoss,
                     metric: MaskScoring,
                     optimizer: optim.Optimizer =None):
    """セグメンテーションのメインルーチン。1エポックの処理を行う

    Args:
        mode (str): モード。'train' or 'val' or 'test'
        epoch (int): 現在のエポック数
        loader (DataLoader): データローダー
        device (torch.device): GPU or CPU
        net (SimpleUNet): ネットワーク
        criterion (nn.CrossEntropyLoss): 損失関数
        metric (MaskScoring): 評価関数
        optimizer (optim.Optimizer, optional): オプティマイザー。訓練のみ指定. Defaults to None.
    """

    if mode == 'train':
        net.train()     # ネットワークを訓練モードに
    else:   # 'val' or 'test' 
        net.eval()      # ネットワークを評価モードに

    display_interval = len(loader) // 10    # 状態表示するミニバッチの間隔

    loss_list = []  # 損失値のリスト
    batch_tp = 0    # データ全体の真陽性のピクセル数
    batch_fp = 0    # データ全体の偽陽性のピクセル数
    batch_fn = 0    # データ全体の偽陰線のピクセル数
    for batch_idx, data in enumerate(loader, start=1):
        images, masks = data  # データローダーから取り出されたタプルを分解
        images, masks = images.to(device), masks.to(device)   # デバイスに送る

        if mode == 'train':
            optimizer.zero_grad()   # 訓練フェーズのとき、パラメータに対する損失の勾配をリセット
            logits = net(images)    # ネットワークの順伝播でロジット値（クラス別のスコア）を算出
            loss = criterion(logits, masks)    # 損失関数にロジットと正解ラベルを入力して損失値を算出
            loss.backward()     # 訓練フェーズのとき、ネットワークの逆伝播でパラメータに対する損失の勾配を算出
            optimizer.step()    # 最適化アルゴリズムでパラメータを更新
        else:
            with torch.no_grad():
                logits = net(images)    # 検証フェーズのとき、勾配を計算せずに高速に順伝播
                loss = criterion(logits, masks)

        loss_list.append(loss.item())   # 損失値をリストに追加

        predictions = torch.argmax(logits, dim=1)   # ロジット値（クラス別のスコア）が最大のクラスを予測クラスとする
        
        minibatch_tp, minibatch_fp, minibatch_fn = metric(predictions, masks)
        batch_tp += minibatch_tp    # ミニバッチの真陽性ピクセル数を加算
        batch_fp += minibatch_fp    # ミニバッチの偽陽性ピクセル数を加算
        batch_fn += minibatch_fn    # ミニバッチの偽陰性ピクセル数を加算
        
        if batch_idx % display_interval == 0:   # 間隔ごとに状態表示
            print(f'{mode} [mini-batches: {batch_idx}, images: {batch_idx * loader.batch_size}] loss: {loss.item():.4f}')
        
        if batch_idx == 1:
            if epoch is not None:
                file_name = f'results/mnist_segmentation/{mode}_epoch{epoch:02}.png'
            else:
                file_name = f'results/mnist_segmentation/{mode}.png'

            # 結果の画像保存
            save_mask_image(images.detach().to('cpu'), predictions, file_name)

    loss_mean = np.mean(loss_list)     # エポック内の損失値の平均

    dump = f'{mode}  loss: {loss_mean:.4f}  TP: {batch_tp}  FP: {batch_fp}  FN: {batch_fn}\n'
    if epoch is not None:
        dump = f'Epoch: {epoch}/' + dump    
    print(dump)
