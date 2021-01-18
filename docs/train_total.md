[toc]

# 训练指南
## 数据集准备
### 验证集
验证集是用于指导训练何时停止,训练结果是否可用的依据,很重要,要求图片选择上尽量贴合具体任务场景,组织形式请参照`HPatchs`数据集,当然,这个可以重写 dataset 来兼容其他数据集的组织形式.

### 训练集
训练集就用一堆 jpg 放在一个文件夹里就行,目前不支持递归,可以重写 dataset 支持

## 配置修改
基础配置文件位于 `../kp2d/configs/base_config.py`
注意设置的图片大小一定要是8的倍数大小
主要配置项看注释,注意在训练时可能需要调整 dataloader 在对图片进行单应性变换时的参数设置,后期可以把这部分参数拉出来

## 训练命令
**训练前请确认各项设置依据配置正确!!!**
### 单机单卡
使用编号7的gpu训练
```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_keypoint_net.py ./kp2d/configs/v4.yaml
```

### 单机多卡
使用编号6,7 2张gpu训练
```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 scripts/train_keypoint_net.py ./kp2d/configs/v4.yaml --launcher pytorch
```
注意这里`nproc_per_node`必须指定,值为GPU数目,`launcher`参数在多卡训练时,必须指定为pytorch 若提示IP或者端口冲突,注意更换`master_port`,
