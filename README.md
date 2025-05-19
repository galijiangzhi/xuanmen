Xuanmen_Net 是一个基于 PyTorch 的先进手语翻译模型，
融合了最先进的mediapipe技术，提供高精度手语翻译能力。本仓库不仅包含推理演示，
还提供完整的训练框架，用户可轻松自定义训练自己的手语翻译模型。

主要特性：

支持手语检测，分类，翻译一体化

支持自定义训练集一键训练

优化后的轻量架构可以在边缘设备上进行快速计算

提供flask部署方案

快速开始：
<details>
<summary>✏️ 环境配置</summary>

### 克隆代码仓库并在Python3.10版本的环境中安装依赖项，请确保已安装cuda12.3及以上版本。推荐使用Python3.10.16+cuda12.3的搭配。
- 支持多级嵌套
- 显示代码、列表、图片等任意内容
```python
#克隆代码仓库
git clone https://github.com/galijiangzhi/xuanmen.git

#导航至项目目录
cd xuanmen

#安装必要的依赖库
pip install -r requirements.txt
```
</details>
<details>
<summary>✏️ 快速运行demo</summary>

![替代文字](./information/model_comparison.png)