# models

`gcu_net.py`：GCU-Net。

- 输入：3×256×256（G、绿度、ROI）。
- 编码器：残差块，通道 32/64/128/256。
- 瓶颈：ASPP（空洞率 3、6、全局池化）。
- 解码器：上采样 + 注意力门控 skip + 残差块。
- 输出：训练时为 `[主输出, 1/2 深监督, 1/4 深监督]`，eval 时只返回主输出。
- 参数量：6.938 M。

`__init__.py` 导出 `GCU_Net`，供 `utils.helpers.get_instance` 按配置实例化。
