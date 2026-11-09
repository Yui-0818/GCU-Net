# utils

训练所需的通用组件。

| 文件 | 内容 |
| --- | --- |
| `losses.py` | BCE、Dice、CE+Dice 等损失；本实验使用 `CE_DiceLoss` |
| `metrics.py` | 二值分割指标：Dice（F1）、IoU、AUC、敏感度、特异度、精确率、准确率 |
| `helpers.py` | 随机种子、配置实例化 |

本实验对 `metrics.py` 做了两处适配：

1. 当一个 batch 内只有背景（无病灶）时，`roc_auc_score` 会抛异常，改为 AUC=0.5；
2. 各项除法加入 `1e-8`，避免空预测时出现除零。
