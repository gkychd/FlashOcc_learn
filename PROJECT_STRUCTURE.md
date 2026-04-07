# FlashOCC 项目结构与模型定义

## 项目概述

FlashOCC (Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin) 是一个基于 **MMDetection3D** 的**语义占用预测 (Semantic Occupancy Prediction)** 模型。该项目实现了:

- **FlashOCC**: 快速高效的占用预测模型
- **Panoptic-FlashOCC**: 全景占用预测 (结合实例中心)

## 项目结构

```
FlashOCC/
├── projects/
│   ├── configs/                    # 配置文件目录
│   │   ├── flashocc/              # FlashOCC 主模型配置
│   │   ├── panoptic-flashocc/     # 全景FlashOCC配置
│   │   └── bevdet_occ/            # BEVDetOcc baseline配置
│   └── mmdet3d_plugin/            # 核心模型代码
│       ├── models/
│       │   ├── detectors/         # 检测器定义 (BEVDetOCC等)
│       │   ├── dense_heads/       # Occupancy预测头
│       │   ├── necks/             # FPN和ViewTransformer
│       │   ├── backbones/         # ResNet, Swin
│       │   ├── losses/            # 损失函数
│       │   └── model_utils/       # 深度估计模块
│       ├── ops/                   # 自定义CUDA算子 (BEV Pool, nearest_assign)
│       ├── datasets/              # 数据集处理
│       └── core/                  # 评估、后处理等
├── tools/                          # 训练/测试/可视化脚本
│   ├── train.py                   # 训练入口
│   ├── test.py                    # 测试入口
│   └── analysis_tools/            # 分析工具
├── doc/                           # 文档
│   ├── install.md                 # 安装指南
│   ├── model_training.md          # 训练文档
│   └── mmdeploy_test.md           # TensorRT部署
└── README.md                      # 项目说明
```

## 模型结构定义位置

模型架构主要定义在以下文件中：

| 组件 | 文件位置 | 描述 |
|------|----------|------|
| **主检测器** | `projects/mmdet3d_plugin/models/detectors/bevdet_occ.py:76-210` | `BEVDetOCC` 类，继承自BEVDet |
| **Occupancy Head** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py` | `BEVOCCHead2D` - 核心的Channel-to-Height模块 |
| **View Transformer** | `projects/mmdet3d_plugin/models/necks/view_transformer.py` | `LSSViewTransformer` - BEV特征提取 |
| **Neck (FPN)** | `projects/mmdet3d_plugin/models/necks/fpn.py` | `CustomFPN` |
| **Backbone** | `projects/mmdet3d_plugin/models/backbones/resnet.py` | ResNet/Swin |

### 主要模型类

- `BEVDetOCC` - 基于BEVDet的占用预测 (单帧)
- `BEVDepthOCC` - 带深度监督的占用预测
- `BEVDepth4DOCC` - 4D时序占用预测
- `BEVStereo4DOCC` - 立体视觉4D占用预测
- `BEVDepthPano` / `BEVDepth4DPano` - 全景占用预测

### 模型数据流

```
输入图像 → img_backbone → img_neck → img_view_transformer → img_bev_encoder → occ_head → 占用预测
                (ResNet)      (FPN)      (LSSViewTransformer)  (BEVEncoder)    (BEVOCCHead2D)
```

## 关键配置文件

- `projects/configs/flashocc/flashocc-r50.py` - 主配置，ResNet50 backbone
- `projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_2e-4.py` - Swin-Base + 4D Stereo配置
- `projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth.py` - 全景占用配置

## 快速开始

### 环境安装

参考 `doc/install.md`

### 训练

```bash
python tools/train.py projects/configs/flashocc/flashocc-r50.py
```

### 测试

```bash
python tools/test.py projects/configs/flashocc/flashocc-r50.py --checkpoint ckpts/xxx.pth
```

## 参考

- [arXiv:2311.12058](https://arxiv.org/abs/2311.12058) - FlashOCC
- [arXiv:2406.10527](https://arxiv.org/abs/2406.10527) - Panoptic-FlashOCC