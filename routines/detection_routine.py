import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from models.simple_pointnet import SimplePointNet
from criteria.pointnet_loss import PointNetLoss
from metrics.detection_metric import BboxScoring
from utils.util import make_bboxes_list
from visualizations.bbox_visualization import save_bbox_image


def run_detection(mode: str,
                  epoch: int,
                  loader: DataLoader,
                  device: torch.device,
                  net: SimplePointNet,
                  criterion: PointNetLoss,
                  metric: BboxScoring,
                  optimizer: optim.Optimizer =None) -> None:
    """物体検出のメインルーチン。1エポックの処理を行う

    Args:
        mode (str): モード。'train' or 'val' or 'test'
        epoch (int): 現在のエポック数
        loader (DataLoader): データローダー
        device (torch.device): GPU or CPU
        net (SimplePointNet): ネットワーク
        criterion (nn.Module): 損失関数
        metric (BboxScoring): 評価関数
        optimizer (optim.Optimizer, optional): オプティマイザー。訓練のみ指定. Defaults to None.
    """

    if mode == 'train':
        net.train()     # ネットワークを訓練モードに
    else:   # val test
        net.eval()      # ネットワークを評価モードに

    display_interval = len(loader) // 10    # 状態表示するミニバッチの間隔

    loss_list = []  # 損失値のリスト
    batch_tp = 0    # データ全体の真陽性のボックス数
    batch_fp = 0    # データ全体の偽陽性のボックス数
    batch_fn = 0    # データ全体の偽陰線のボックス数
    for batch_idx, data in enumerate(loader, start=1):
        # データローダーから取り出されたタプルを分解
        images, target_bboxes_list, target_labels_list, \
        target_keypoints, target_offsets, target_sizes = data

        # デバイスに送る
        images = images.to(device)
        target_keypoints = target_keypoints.to(device)
        target_offsets = target_offsets.to(device)
        target_sizes = target_sizes.to(device)   

        if mode == 'train':
            optimizer.zero_grad()   # 訓練フェーズのとき、パラメータに対する損失の勾配をリセット

            # ネットワークの順伝播でロジット値（クラス別のスコア）を算出
            predicted_keypoints, predicted_offsets, predicted_sizes = net(images)    

            # 損失関数にロジットと正解ラベルを入力して損失値を算出
            loss = criterion(predicted_keypoints, predicted_offsets, predicted_sizes,
                             target_keypoints, target_offsets, target_sizes)

            loss.backward()     # 訓練フェーズのとき、ネットワークの逆伝播でパラメータに対する損失の勾配を算出
            optimizer.step()    # 最適化アルゴリズムでパラメータを更新
        
        else:   # val/test
            with torch.no_grad():
                # 検証フェーズのとき、勾配を計算せずに高速に順伝播
                predicted_keypoints, predicted_offsets, predicted_sizes = net(images)

                loss = criterion(predicted_keypoints, predicted_offsets, predicted_sizes,
                                 target_keypoints, target_offsets, target_sizes)
        
        loss_list.append(loss.item())   # 損失値をリストに追加

        predicted_bboxes_list, predicted_labels_list = make_bboxes_list(predicted_keypoints,
                                                                        predicted_offsets,
                                                                        predicted_sizes,
                                                                        k=3)    # 3個までのボックスを推定
        
        score = metric(predicted_bboxes_list, predicted_labels_list,
                       target_bboxes_list, target_labels_list)

        minibatch_tp, minibatch_fp, minibatch_fn = score
        batch_tp += minibatch_tp    # ミニバッチの真陽性ボックス数を加算
        batch_fp += minibatch_fp    # ミニバッチの偽陽性ボックス数を加算
        batch_fn += minibatch_fn    # ミニバッチの偽陰性ボックス数を加算
        
        if batch_idx % display_interval == 0:   # 一定間隔ごとに状態表示
            print(f'{mode} [mini-batches: {batch_idx}, images: {batch_idx * loader.batch_size}] loss: {loss.item():.4f}')
        
        if batch_idx == 1:
            if epoch is not None:
                file_name = f'results/mnist_detection/{mode}_epoch{epoch:02}.png'
            else:
                file_name = f'results/mnist_detection/{mode}.png'

            # 結果の画像保存
            save_bbox_image(images.detach().to('cpu'),
                            predicted_bboxes_list, predicted_labels_list,
                            target_bboxes_list, target_labels_list,
                            file_name)

    loss_mean = np.mean(loss_list)     # エポック内のミニバッチ損失値の平均

    dump = f'{mode}  loss: {loss_mean:.4f}  TP: {batch_tp}  FP: {batch_fp}  FN: {batch_fn}\n'
    if epoch is not None:
        dump = f'Epoch: {epoch}/' + dump    
    print(dump)