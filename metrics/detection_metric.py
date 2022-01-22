from typing import Tuple
import torch
from torch import Tensor
from torchvision.ops import box_iou


class BboxScoring:
    """物体検出のTP・FP・FNを評価するクラス
    """
    def __init__(self, iou_threshold: float =0.75) -> None:
        """初期化メソッド

        Args:
            iou_threshold (float, optional): 判定のIoU閾値. Defaults to 0.75.
        """
        self.iou_threshold = iou_threshold
    
    def _classify_bboxes(self,
                         predicted_bboxes: Tensor,
                         target_bboxes: Tensor) -> Tuple[float, float, float]:
        """画像1枚に対する真陽性、偽陽性、偽陰性を計算する

        Args:
            predicted_bboxes (Tensor): 推論ボックス。Tensor形状は[N, 4]
            target_bboxes (Tensor): 正解ボックス。Tensor形状は[M, 4]

        Returns:
            Tuple[float, float, float]: 画像1枚の真陽性、偽陽性、偽陰性
        """

        N = len(predicted_bboxes)   # 推論ボックスの個数
        M = len(target_bboxes)  # 正解ボックスの個数
        
        if N == 0 or M == 0:
            image_tp = 0    # 推論ボックスと正解ボックスのいずれかがないなら、真陽性はない
        else:
            iou_matrix = box_iou(predicted_bboxes, target_bboxes)   # IoUマトリックス。Tensor形状は[N, M]

            sort_indices = iou_matrix.max(dim=1)[0].argsort(descending=True)    # IoU降順のインデックス
            iou_matrix = iou_matrix[sort_indices]   # 降順にIoUマトリックスをソート

            # 対応した箇所がTrueになるGTと推論の対応表を作成する。Tensor形状は[N, M]
            correspondence = torch.zeros_like(iou_matrix, dtype=torch.long)     

            # 未割り当ての正解ボックスがTrueになる配列を作成する。Tensor形状は[M]
            unassigned_mask = torch.ones(M, dtype=torch.bool) 
            
            # 閾値を超えた箇所がTrueになる対応表を作成する。Tensor形状は[N, M]
            thresholded_mask = iou_matrix > self.iou_threshold

            for n in range(N):  # ソートしたIoUマトリックスを行ごとに順に処理する
                # 未割り当て、かつ、閾値を超えた箇所がTrueの配列。Tensor形状は[M]
                enabled_mask = torch.logical_and(unassigned_mask, thresholded_mask[n])

                if enabled_mask.any() == False:
                    continue    # 未割り当て、かつ、閾値を超えた箇所がない行はスキップ

                enabled_iou = enabled_mask * iou_matrix[n]
                m = torch.argmax(enabled_iou)   # 未割り当て、かつ、閾値を超えた中でIoUが最大の正解ボックス
                
                # 対応表に割り当てる
                correspondence[n, m] = 1

                # 使用した正解ボックスをunassiginedから除去する
                unassigned_mask[m] = False

            image_tp = correspondence.sum().item()  # 対応表のTrueの個数が真陽性に等しい

        image_fp = N - image_tp     # 推論ボックスで真陽性にならなかったものが偽陽性
        image_fn = M - image_tp     # 正解ボックスで真陽性にならなかったものが偽陰性

        return image_tp, image_fp, image_fn

    def __call__(self,
                 predicted_bboxes_list: Tuple[Tensor, ...],
                 predicted_labels_list: Tuple[Tensor, ...],
                 target_bboxes_list: Tuple[Tensor, ...],
                 target_labels_list: Tuple[Tensor, ...]) -> Tuple[float, float, float]:
        """ミニバッチの真陽性、偽陽性、偽陰性を算出する

        Args:
            predicted_bboxes_list (Tuple[Tensor, ...]): 推論ボックスのタプル
            predicted_labels_list (Tuple[Tensor, ...]): 推論ラベルのタプル
            target_bboxes_list (Tuple[Tensor, ...]): 正解ボックスのタプル
            target_labels_list (Tuple[Tensor, ...]): 正解ラベルのタプル

        Returns:
            Tuple[float, float, float]: ミニバッチの真陽性、偽陽性、偽陰性
        """
        minibatch_tp = 0    # ミニバッチのTPの個数
        minibatch_fp = 0    # ミニバッチのFPの個数
        minibatch_fn = 0    # ミニバッチのFNの個数
        
        for predicted_bboxes, predicted_labels, target_bboxes, target_labels in zip(predicted_bboxes_list, predicted_labels_list, target_bboxes_list, target_labels_list):
            # ユニークなラベルの配列を作る
            if len(predicted_labels) == 0:
                unique_labels = target_labels.unique()
            else:
                unique_labels = torch.cat([predicted_labels, target_labels]).unique()

            for unique_label in unique_labels:
                # ユニークラベルな推論ボックス
                if len(predicted_labels) == 0:
                    p_bboxes = []
                else:
                    p_bboxes = predicted_bboxes[predicted_labels == unique_label]
    
                # ユニークラベルな正解ボックス
                t_bboxes = target_bboxes[target_labels == unique_label]

                # 画像の真陽性、偽陽性、偽陰性
                image_tp, image_fp, image_fn = self._classify_bboxes(p_bboxes, t_bboxes)

                minibatch_tp += image_tp
                minibatch_fp += image_fp
                minibatch_fn += image_fn
        
        return minibatch_tp, minibatch_fp, minibatch_fn


if __name__ == '__main__':
    """テストコード
    """
    batch_size = 100
    max_num_boxes = 5
    num_classes = 10
    
    predicted_num_boxes = torch.randint(low=0, high=max_num_boxes+1, size=[batch_size])
    target_num_boxes = torch.randint(low=1, high=max_num_boxes+1, size=[batch_size])

    predicted_bboxes_list = []
    predicted_labels_list = []
    target_bboxes_list = []
    target_labels_list = []

    for p_num_boxes, t_num_boxes in zip(predicted_num_boxes, target_num_boxes):
        p_xmin_ymin = torch.rand(size=[p_num_boxes.item(), 2])
        p_width_height = 1. + torch.rand(size=[p_num_boxes.item(), 2])
        predicted_bboxes = torch.cat([p_xmin_ymin, p_xmin_ymin+p_width_height], dim=1)

        predicted_labels = torch.randint(low=0, high=num_classes, size=[p_num_boxes.item()])

        t_xmin_ymin = torch.rand(size=[t_num_boxes.item(), 2])
        t_width_height = 1. + torch.rand(size=[t_num_boxes.item(), 2])
        target_bboxes = torch.cat([t_xmin_ymin, t_xmin_ymin+t_width_height], dim=1)

        target_labels = torch.randint(low=0, high=num_classes, size=[t_num_boxes.item()])

        predicted_labels_list.append(predicted_labels)
        predicted_bboxes_list.append(predicted_bboxes)

        target_labels_list.append(target_labels)
        target_bboxes_list.append(target_bboxes)

    metrics = BboxScoring(iou_threshold=0.75)
    minibatch_tp, minibatch_fp, minibatch_fn = metrics(predicted_bboxes_list,
                                                       predicted_labels_list,
                                                       target_bboxes_list,
                                                       target_labels_list)
    
    print(f'th=0.75  TP: {minibatch_tp}  FP: {minibatch_fp}  FN: {minibatch_fn}')

    metrics = BboxScoring(iou_threshold=0.5)
    minibatch_tp, minibatch_fp, minibatch_fn = metrics(predicted_bboxes_list,
                                                       predicted_labels_list,
                                                       target_bboxes_list,
                                                       target_labels_list)
    
    print(f'th=0.5  TP: {minibatch_tp}  FP: {minibatch_fp}  FN: {minibatch_fn}')

    metrics = BboxScoring(iou_threshold=0.25)
    minibatch_tp, minibatch_fp, minibatch_fn = metrics(predicted_bboxes_list,
                                                       predicted_labels_list,
                                                       target_bboxes_list,
                                                       target_labels_list)
    
    print(f'th=0.25  TP: {minibatch_tp}  FP: {minibatch_fp}  FN: {minibatch_fn}')