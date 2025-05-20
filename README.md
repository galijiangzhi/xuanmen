Xuanmen_Net 是一个基于 PyTorch 的先进手语翻译模型，
融合了最先进的mediapipe技术，提供高精度手语翻译能力。本仓库不仅包含推理演示，
还提供完整的训练框架，用户可轻松自定义训练自己的手语翻译模型。

主要特性：

* 支持手语检测，分类，翻译一体化
* 支持自定义训练集一键训练
* 优化后的轻量架构可以在边缘设备上进行快速计算
* 提供flask部署方案

快速开始：
<details open>
<summary><h3>📖 环境配置</h3></summary>

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
<summary><h3>📖 快速运行demo</h3></summary>

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
<summary><strong style="font-size: 30px">📖 训练模型</strong></summary>

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
<details open>
<summary><strong style="font-size: 26px">✏️ 1.创建数据集</strong></summary>

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
<details open>
<summary><strong style="font-size: 26px">✏️ 2.模型参数选择</strong></summary>

设置一套参数来启动训练过程，对于不同的应用场景选择不同的参数可以有效提高系统的结果，
Xuanmen_Net在实验阶段测试了一系列模型参数，每种模型都能在速度和准确性之前取得不同的平衡。

## 🎯核心参数说明

|参数名称	|可选范围	|影响说明|
| :---: | :---: | :---: |
|聚类数 (K)	|20-100	|值越大手势分类越细，但超过峰值后收益递减（实验数据集阈值为50）
|嵌入维度	|128/256/512	|影响特征表达能力，512维在多数场景最优但计算量较大
|隐藏层维度	|通常与嵌入维度相同	|决定LSTM记忆容量，与嵌入维度同步调整
|双手处理	|合并/独立	|独立处理精度高5-8%，但计算量翻倍
|抽帧间隔	|1-7帧	|间隔1~3时最佳（实验数据集间隔3效果最好计算量减少67%且BLEU提升0.8%）

## 💻 场景化推荐配置

|     配置类型      | 聚类数 | 嵌入维度 | 隐藏层 | 双手处理 | 抽帧间隔 | 预期效果        |
|:-------------:|:--------:|:----------:|:--------:|:----------:|:----------:|:-------------:|
| 移动端APP (轻量型)  | 40     | 128      | 128    | 合并     | 5        | BLEU2≈0.64  |
| 移动端APP (平衡型)  | 50     | 256      | 256    | 独立     | 3        | BLEU2≈0.66  |
|  桌面级应用 (高性能)  | 80     | 512      | 512    | 独立     | 1        | BLEU2≈0.669 |


## ⚙️ 高级调参建议

### 数据量较小时：
* 启用多头注意力 (multi-head=True)，适当降低聚类数 (K=30-40)，可提升小数据场景效果11%

### 实时性要求高时：

```python
# 在config.yaml中调整
sampling_interval: 5  # 抽帧间隔
hands_model: True     # 双手合并
```
### 精度优先时：
```python
多头: True  
embedding_dim: 512
cluster_num: 80
```
</details>
<details open>

<summary><strong style="font-size: 26px">✏️ 3.整理目录</strong></summary>

## 构建您的数据集目录

推荐的目录结构为

```
xuanmen(git代码仓库根路径）
└── SLR_dataset/  # 数据根路径
    ├── color/   #视频数据路径
    │   ├── 000000/  # 存放第一类语义的视频，如'他的同学是教师‘，内部视频文件名随意
    │   │   ├── 1.avi
    │   │   ├── 2.avi
    │   │   └── ...
    │   ├── 000001/  # 存放第二类语义的视频，如'我的毛巾是干的‘，内部视频文件名随意
    │   │   ├── 1.avi
    │   │   ├── 2.avi
    │   │   └── ...
    │   └── ...
    │
    ├── kme_seq/   #kmean聚类后的序列文件夹
    │
    ├── model/   #模型存放文件夹
    │
    ├── seq_txt/   #视频序列化存储文件夹
    │
    └──corpus.txt   # label——folder映射文件

```
</details>
<details open>

<summary><strong style="font-size: 26px">✏️ 4.修改配置文件</strong></summary>

## Xuanmen_Net项目需要配置合理的参数用于开发自定义手语识别模型，正确编写参数文件是关键

### 配置文件概述了数据集的结构，序列翻译模型定义的，手势提取模型的结构以及关键文件的路径，并且还有一个流程表用于控制项目只运行某些模块。

默认的示例配置文件 **scripts/config/config.yaml** 文件包括：

* model: 包含双手模式,取样区间，手势类别，序列长度，多头模式，词向量维度，隐藏层维度，n_layers，以及多头的数量。
* train: 包括训练批次和批次大小
* mediapipe: 手势提取网络的关键参数，与mediapipe类的定义一致
* 流程:控制每个步骤是否要进行运行
* 数据清洗: 控制特征提取后的文件过滤，删除零值过高的数据以保证数据的高质量
* 计算参数:控制最大进程数
* path: 记录关键路径，如数据集视频路径，数据序列路径，模型路径等

以下是默认配置文件config.yaml[(在GitHub上查看)](https://github.com/galijiangzhi/xuanmen/blob/main/scripts/config/config.yaml)，其中标注带双星号（**）的为需要注意修改项:

```yaml
model:
  双手合并: True  #** true为双手合并表示，false为双手分离表示
  取样区间: 1     #** 该参数用于控制每多少帧取一帧，1表示每帧都取，2表示每2帧取1帧
  kmeans类别: 40  #** 手势分类的类别数
  kmeans训练批次大小: 5000  #训练kmeans时的批次大小，根据内存大小设置即可，5000为内存16g的参考值
  不抽帧双手分离seq最大长度: 800 #** 设置默认情况下单个手势的最长帧率，计算方式为 最长视频时间*帧率*2
  多头: False #** 是否启用多头网络
  词向量维度: 512 #** 设置词向量维度
  LSTM隐藏层维度: 512 #** 设置隐藏层维度
  n_layers: 1 #** 设置网络深度
  num_heads: 8 #设置多头的数量，如果不使用多头则掠过

train:
  epoch: 10 #** 训练次数
  batch_size: 128 #** 批次大小

mediapipe:
  static_image_mode: False # 视频流模式（跟踪优先）
  max_num_hands: 2  # 最多检测2只手
  model_complexity: 1  # 中等复杂度模型
  min_detection_confidence: 0.4  # 检测置信度阈值较高
  min_tracking_confidence: 0.3  # 跟踪置信度阈值

流程:
  video2txt: False #** 流程控制：是否要进行图像特征提取，将每个视频提取为数字序列 每个文件形状为 帧率*2*63（双手分离） 或者 帧率*126(双手合并）
  数据零值比例过滤: False #** 流程控制：是否要对提取到的数字序列进行过滤，删除零值过高的数据
  整合合理数据: False #** 流程控制：是否要将所有的数字序列合并到一个文件夹，该流程是kmeans训练的必要流程
  kmeans模型训练: False #** 流程控制：是否要训练kmeans模型
  kmeans对原数据进行处理: False #** 流程控制：是否要使用kmeans模型对数字序列进行降维
  xuanmen网络训练: True #** 流程控制：是否要训练‘翻译网络’


数据清洗:
  零值比例阈值: 0.74 #零值比例过滤参数，非零值低于该值的数据将被定义为低质量数据被删除

计算参数:
  max_workers: 6 #程序最大进程数量

path:
  dataset_video_path: './SLR_dataset/color' #** 数据集视频地址
  dataset_seq_path: './SLR_dataset/seq_txt' #** 数据集特征提取之后的保存地址
  双手合并全部数据npy_path: './SLR_dataset/准确率正常的全部数据_双手合并.npy' #** 双手合并的全部数据
  双手分离全部数据npy_path: './SLR_dataset/准确率正常的全部数据_双手分离.npy' #** 双手分离的全部数据
  kmeans_root_path: './SLR_dataset/model/kmeans' #** kmeans模型保存的根目录
  kmeseq_root_path: './SLR_dataset/kme_seq' #** kmeans降维之后的数据的保存目录
  数据文件夹标签path: './SLR_dataset/corpus.txt' #** label-folder映射文件
  xuanmen_model_root_path: './SLR_dataset/model/' #** 模型保存根目录
log:
  video_process_log: "./scripts/config/video_processed.log" #单一视频mediapipe处理可视化保存路径

```
</details>
<details open>

<summary><strong style="font-size: 26px">✏️ 5.运行训练程序</strong></summary>
## 修改配置文件后，运行以下代码执行模型训练程序

```python
cd xuanmen #定位到Git代码仓库根目录
python main.py #执行训练程序 
```
</details>
</details>

![替代文字](./information/model_comparison.png)