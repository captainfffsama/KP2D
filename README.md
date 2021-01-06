# Neural Outlier Rejection for Self-Supervised Keypoint Learning

**This project just fork from https://github.com/TRI-ML/KP2D ,and remove docker and horovod rely. you can use conda test this code**   
**本算法的配置文件在`./kp2d/configs/base_config.py`,使用之前请修改其中对于的数据集路径等**

## Overview
![diagram_architecture.png](https://dev.chiebot.com:10443/images/2021/01/06/diagram_architecture.png)
- **IO-Net:** A novel proxy task for the self-supervision of keypoint description. 
- **KeyPointNet:**  An improved keypoint-network architecture that is especially amenable to robust keypoint detection and description.

[**[Full paper]**](https://openreview.net/pdf?id=Skx82ySYPH)

### Setting up your environment

change the `prefix` in the *environment.yaml*,then use 
```python
conda env create -f environment.yaml
```
install conda environment.

Code have been test in Ubuntu 18.04, GTX2070,CUDA 10.01

#### Requiement
pytorch > 1.6  
opencv  
pillow  
yacs  
termcolor  
tqdm  
wandb  

## Install kp2d 
现在网络结构模块 kp2d可以独立安装了
```bash
pip install -e .
```

### Data

Download the HPatches dataset for evaluation:

```bash
cd /data/datasets/kp2d/
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
mv hpatches-sequences-release HPatches
```

Download the COCO dataset for training:
```bash
mkdir -p /data/datasets/kp2d/coco/ && cd /data/datasets/kp2d/coco/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

### Training

To train a model run:

```bash
python scripts/train_keypoint_net.py kp2d/configs/v4.yaml
```

以上训练是单卡的,多卡DDP的尚未测试.

### Pre-trained models:

Download the pre-trained models from [here](https://tri-ml-public.s3.amazonaws.com/github/kp2d/models/pretrained_models.tar.gz) and place them in `/data/models/kp2d/`

To evaluate any of the models, simply run:

```bash
python scripts/eval_keypoint_net.py --pretrained_model /data/models/kp2d/v4.ckpt --input /data/datasets/kp2d/HPatches/
```

Evaluation for **`(320, 240)`**:

| Model	| Repeatability |	Localization |	C1 |	C3 | 	C5 |	MScore |
|---|---|---|---|---|---|---|
| V0*|	0.644 | 1.087 | 0.459 |	0.816 |	0.888 |	0.518 |
| V1*|	0.678 |	0.98  | 0.453 |	0.828 |	0.905 | 0.552 |
| V2*|	0.679 |	0.942 |	0.534 |	0.86  |	0.914 |	0.573 |
| V3|	0.685 |	0.885 |	0.602 |	0.836 |	0.886 |	0.52  |
| V4|	0.687 |	0.892 |	0.593 |	0.867 |	0.91  |	0.546 |


Evaluation for **`(640, 480)`**:

| Model	| Repeatability |	Localization |	C1 |	C3 | 	C5 |	MScore |
|---|---|---|---|---|---|---|
| V0*|	0.633 | 1.157 | 0.45  |	0.81  |	0.89  |	0.486 |
| V1*|	0.673 |	1.049 | 0.464 |	0.817 |	0.895 | 0.519 |
| V2*|	0.68  |	1.008 |	0.51  |	0.855 |	0.921 |	0.544 |
| V3|	0.682 |	0.972 |	0.55  |	0.812 |	0.883 |	0.486 |
| V4|	0.684 |	0.972 |	0.566 |	0.84  |	0.9   |	0.511 |

*-these models were trained again after submission - the numbers deviate slightly from the paper, however the same trends can be observed.


### Over-fitting Examples

These examples show the model over-fitting on single images. For each image, we show the original frame with detected keypoints (left), the score map (center) and the random crop used for training (right). As training progresses, the model learns to detect salient regions in the images.

- **Toy example:**
<p align="center">
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/v1-min.gif" alt="Target Frame" width="230" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/h1-min.gif" alt="Heatmap" width="230" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/w1-min.gif" alt="Source Frame" width="230" />
</p>

- **TRI example:**
<p align="center">
  <img src="https://dev.chiebot.com:10443/images/2021/01/05/compressed_v2.gif" alt="Target Frame" width="230" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/compressed_h2-min.gif" alt="Heatmap" width="230" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/compressed_w2-min.gif" alt="Source Frame" width="230" />
</p>

### Qualatitive Results
- **Illumination Cases:**

<p align="center">
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/l1.png" alt="Illumination case(1)" width="600" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/l2.png" alt="Illumination case(2)" width="600" />
</p>

- **Perspective Cases:**
<p align="center">
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/p1.png" alt="Perspective case(1)" width="600" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/p2.png" alt="Perspective case(2)" width="600" />
</p>

- **Rotation Cases:**
<p align="center">
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/r1.png" alt="Rotation case(1)" width="600" />
  <img src="https://dev.chiebot.com:10443/images/2021/01/06/r2.png" alt="Rotation case(2)" width="600" />
</p>

### License

The source code is released under the [MIT license](LICENSE.md).


### Citation
Please use the following citation when referencing our work:
```
@inproceedings{
tang2020neural,
title={Neural Outlier Rejection for Self-Supervised Keypoint Learning},
author={Jiexiong Tang and Hanme Kim and Vitor Guizilini and Sudeep Pillai and Rares Ambrus},
booktitle={International Conference on Learning Representations},
year={2020}
}
```
