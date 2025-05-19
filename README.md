Xuanmen_Net 是一个基于 PyTorch 的先进手语翻译模型，
融合了最先进的mediapipe技术，提供高精度手语翻译能力。本仓库不仅包含推理演示，
还提供完整的训练框架，用户可轻松自定义训练自己的手语翻译模型。

主要特性：

支持手语检测，分类，翻译一体化

支持自定义训练集一键训练

优化后的轻量架构可以在边缘设备上进行快速计算

提供flask部署方案

快速开始：
<details open>
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
<details open>
<summary>✏️ 快速运行demo</summary>

### 我们提供了建议的demo用于展示结果，请确保已完成环境配置后执行以下代码运行示例程序

```python
#确保当前路径为git项目根文件夹
pwd

#导航至demo程序目录
cd demo/flask

#运行demo程序
python main.py
```

等待后端程序启动之后，通过浏览器访问 http://127.0.0.1:5000 可以打开demo测试页面，前端结构[如图所示（点击查看）](https://github.com/galijiangzhi/xuanmen/blob/main/information/demo_%E5%89%8D%E7%AB%AF%E7%A4%BA%E6%84%8F%E5%9B%BE.png)。

模型选择建议使用 km40-emb256-hid256_双手分离_抽帧1，
同时我们提供了一些测试视频，测试视频根路径为 'xuanmen/demo/demo_data'，
这些测试视频会在克隆代码仓库的时候一并克隆到本地。

</details>
<details open>
<summary>✏️ 训练模型</summary>

### 本指南介绍如何使用 Xuanmen_Net模型 训练您自己的自定义数据集。训练自定义模型是定制手语翻译解决方案的基本步骤。

## 开始之前

首先，确保您已建立必要的环境。克隆Xuanmen_Net代码仓库，配置python3.10环境 并从 requirements.txt 中安装必要的依赖库，
同时训练神经网络的gpu环境也是必要的。推荐使用Python3.10.16+cuda12.3的搭配，pytorch版本在requirements.txt有定义，
在安装依赖库列表时会自动安装搭配推荐环境使用的pytorch环境。

```python
git clone https://github.com/galijiangzhi/xuanmen.git
cd xuanmen
pip install -r requirements.txt
```

开发自定义手语识别模型是一个复杂过程：
* 收集和整理视频数据：收集与特定任务相关的高质量手语视频数据。
* 整理数据：按照手语的含义对视频数据进行分类，并创建folder-label对照表，该表用于进行’文件夹名‘-'手语含义'映射。
* 手语序列提取：将全部的手语数据进行汇总，训练kmeans聚类网络对对高维手部轨迹进行编码，获得紧凑的类别化表示
* 训练序列翻译网络：训练seq2seq网络，将类别化表示的手部动作序列翻译为目标语言
* 部署与预测：利用训练后的模型对未见过的手部序列进行推理
* 收集边缘案例：找出模型表现不佳的情况，将类似数据添加到数据集中，以提高模型性能，循环训练。

我们为训练自定义手语模型过程提供了的代码，包括手部序列提取，kmeans模型训练，手部序列类别化，seq2seq网络训练等。

# 1.创建数据集

Xuanmen_Net模型需要整理好的数据来学习手语的特征，正确地整理手语视频数据是模型训练的关键。

用户需按以下规则整理手语数据：

```python
dataset/
├── 000000/         # 存放第一类语义的视频，如'他的同学是教师‘，内部视频文件名随意
│   ├── 1.mp4    
│   ├── 2.mp4    
│   ├── 3.mp4    
│   └── ...
├── 000001/         # 存放第一类语义的视频，如'我买了苹果‘，内部视频文件名随意
│   ├── 1.mp4    
│   ├── 2.mp4    
│   ├── 3.mp4    
│   └── ...
└── corpus.txt      # 标签映射文件
```

标签映射文件格式:

```python
000000 他的同学是警察
000001 他妈妈的同学是公务员
000002 我的爸爸是商人
000003 他哥哥的目标是解放军
```

## 标签映射文件字段说明：

* 前六位数字：必须与视频父文件夹名严格匹配
* 文本内容：描述对应文件夹下所有视频的含义

## 数据集建议

* 每个标签至少50个视频样本
* 单视频时长5-10秒
* 手部动作不宜过快，最佳速度为12cm/s-18cm/s

</details>

![替代文字](./information/model_comparison.png)