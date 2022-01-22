# MNIST Semantic Segmentation and Obkect Detection

PyTorchを使ったシンプルなセマンティックセグメンテーションと物体検出のコードです。

データセットはMNISTを加工して使用します。

![](title_a.png)![](title_b.png)

## requirements

```
albumentations         1.1.0 
jupyter                1.0.0
matplotlib             3.5.1
notebook               6.4.6
numpy                  1.22.0
Pillow                 9.0.0
scikit-image           0.19.1
torch                  1.10.1
torchvision            0.11.2
```

## Semantic Segmentation

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

## Object Detection

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