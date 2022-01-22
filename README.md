# MNIST Semantic Segmentation and Object Detection

PyTorchを使ったシンプルなセマンティックセグメンテーションと物体検出のコードです。

学習するデータセットはMNISTを加工して使用します。

<img src="title_a.png" width="400px"> <img src="title_b.png" width="400px">

## requirements

```
albumentations >= 1.1.0 
jupyter >= 1.0.0
matplotlib >= 3.5.1
torch >= 1.10.1
torchvision >= 0.11.2
```

## セマンティックセグメンテーション

### 推論例

![](results/mnist_segmentation/test.png)

### 学習

```
python learning_segmentation.py
```

### 推論

```
python inference_segmentation.py
```

### 学習と推論のノートブック

```
mnist_segmentation.ipynb
```

## 物体検出

### 推論例

![](results/mnist_detection/test.png)

### 学習

```
python learning_detection.py
```

### 推論

```
python inference_detection.py
```

### 学習と推論のノートブック

```
mnist_detection.ipynb
```
