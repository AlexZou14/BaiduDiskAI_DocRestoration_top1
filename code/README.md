# 百度网盘AI大赛-模糊文档图像恢复比赛第1名方案

非常幸运获得了百度网盘大赛，[模糊文档图像恢复比赛](https://aistudio.baidu.com/aistudio/competition/detail/349/0/leaderboard)AB榜都第一的好成绩！

特别感谢，txyugood提供的[Restormer_paddle](https://github.com/txyugood/Restormer_Paddle)的代码，有效的帮助我们构建了baseline和调参调优。

# 一、赛题分析
此次大赛主题结合日常生活常见情景展开，当使用移动设备扫描获取文档的过程中，很多是文字、字母和数字等符号内容。通过拍摄截取等方式获取文档，就非常有可能导致内容模糊、噪音叠加等问题的发生，使得无法实际发挥作用。期望同学们通过计算机技术，帮助人们将模糊失焦的文档恢复清晰，提高使用便捷度和效率。

# 二、 数据分析
- 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共1000对样本，A榜测试集共200个样本，B榜测试集共200个样本；抽取一部分数据如图：
![](https://ai-studio-static-online.cdn.bcebos.com/347f2446875d40d9968da672f483976b7f1e78c7dc594ffab21a45f7c82e792c)
![](https://ai-studio-static-online.cdn.bcebos.com/978ef62d463343f6b4a5e95d81f34cbcc5050a42ee8f460484bdc7de0d5b3937)
![](https://ai-studio-static-online.cdn.bcebos.com/1e1986723a4b413d973722ccbea4a4d3606c22f05d8d4e9fad2582a123d6f39e)

- blur_image 为模糊的文档图像数据，gt_image 为非模糊的真值数据（仅有训练集数据提供gt_image ，A榜测试集、B榜测试集数据均不提供gt_image）；
- blur_image 与 gt_image 中的图片根据图片名称一一对应。
- 进一步分析训练数据，计算训练数据的平均大小和测试数据的平均大小，我们会发现训练数据的图像大小为测试数据大小的大约2倍左右。因此训练数据和测试数据不在一个domain，因此训练时我们针对训练图像进行了下采样两倍的处理。

# 二、评价标准
评价指标为 PSNR 和 MSSSIM；

用于评价的机器环境仅提供两种框架模型运行环境：paddlepaddle 和 onnxruntime，其他框架模型可转换为
上述两种框架的模型；

机器配置：V100，显存16G，内存10G；

单张图片耗时>1.2s，决赛中的性能分数记0分。

由评价标准可知，不能使用大模型。

# 三、算法介绍

Baseline的选择：针对图像去模糊这个任务，我们首先查询了paper with code网站，找到了目前优异的几个模型：MADANet、NAFNet以及Restormer。根据paper with code的信息我们可以了解到MADANet采用了额外数据进行训练，并且这个赛道数据量并不是很多。因此我们采用了NAFNet作为我们此次的baseline。
网络主体架构为UNet，如图：

![](https://ai-studio-static-online.cdn.bcebos.com/c2ced773bf7b4d9db72ed48c4b92999964dd4198103c463282fa46a62cd9d319)

其中Encoder和Decoder采用NAFBlock:

![](https://ai-studio-static-online.cdn.bcebos.com/a8b2262aaa144f8ea8c9c69087b0f93d988fc976e70e4e8ca61b2c0e88f274df)


由于比赛推理速度需要每张图像1.2s之内，我们对原有的NAFNet进行了通道缩减和减少深度的操作来使得A榜测试图像推理能达到0.8305秒每张图，图中各个进行下采样的block数量设置为1，最深层的block数量设置为10，在单卡V100上训练600000次迭代。为了进一步提升网络性能，我们还进一步采用了Test-time Local Converter (TLC)。主要参考了[Improving Image Restoration by Revisiting Global Information Aggregation论文](https://arxiv.org/abs/2112.04491)

# 四、数据增强与清洗

### 数据划分
官方给的数据为1000张图像，我们将最后的100张图像作为测试集，前面900张作为训练集

### 数据增广
- 我们对训练集进行了裁剪为1024大小的patch。
- 在训练模型的过程中，为了进一步充分利用所有的数据，我们采用了图像翻转，图像旋转等操作来增广数据集。
- 为了加快训练过程，我们还采用了随机裁剪策略。

### 数据清洗
我们发现在裁剪过程中，会裁剪出很多空白图块以及一些信息很少的图块。这些图块并不会帮助网络学习到对应去模糊的知识，因此我们针对这样的情况进行了数据清洗。由于裁剪后的数据有7w+的数据量，筛选非常困难，并且在Aistudio上进行训练的话，需要不断重新导入数据。因此，我们考虑到计算每个图块的平均梯度来度量每个图块包含复杂纹理的多少，并且进行统计。如下所示：

![](https://ai-studio-static-online.cdn.bcebos.com/d7f532be7a444bbfafd4b19c50787c2865b97e7032604963bc511060aaea38b6)


从统计图中我们不难看出，大多数图块都分布在20左右，低于10的图块较少。我们在进一步比对裁剪后的数据，发现图像平均梯度小于10的图块大多都是空白以及一些纹理少的图块。因此，我们直接将平均梯度大于10的图块地址写入txt文件中，通过读取txt文件中的地址来读取数据，然后构建训练集。

# 五、训练细节
### 训练配置
- 总迭代数：600000 iteration
- 我们采用了渐进训练的方式，来加速训练过程。分别采用batch size为8和patch size为192来进行训练184000次迭代，batch size为5和patch size为256来进行训练128000次迭代，batch size为4和patch size为320来进行训练96000次迭代，batch size为2和patch size为384来进行训练72000次迭代，batch size为1和patch size为512来进行训练72000次迭代，batch size为1和patch size为1024来进行训练48000次迭代。
- 我们采用了余弦退火的学习率策略来优化网络，在184000次迭代以及416000迭代处进行初始学习率
- 学习率我们采用2e-4，优化器为AdamW

### 训练分为阶段：
- 第一阶段损失函数为L1Loss
- 第二阶段损失函数使用Charbonnier Loss+MS_SSIMLoss
- 最后，在全量数据进行finetuning

# 六、测试细节
由于测试图像大多为2K图像，直接预测图像会导致显存不足的问题，因此我们采用了裁剪预测的方式来进行，将每张输入图像在输入网络前平均裁剪成大小一致的4块，不能被2整除的图像进行padding再裁剪，最后通过网络预测出四个图像块，然后再通过拼接操作，拼接成输出图像来输出。

# 七、代码结构
- config: 配置文件
- data: 定义数据增强函数
- models: 定义网络模型以及损失函数
- metrics: 定义评价指标函数
- utils: 定义其他函数
- dataset.py: 定义数据集
- output: 模型训练输出文件夹
- data_preparation.py: 训练数据裁剪脚本
- data_preparation_all.py: 全量训练数据裁剪脚本
- generate_meta_info.py: 数据筛选脚本
- generate_meta_info_all.py: 全量数据筛选脚本
- predict.py: 预测脚本
- train.py: 训练脚本

# 八、上分策略

![](https://ai-studio-static-online.cdn.bcebos.com/2bffa4513a1f498da0945379160fa87e420dd753765d4604bc5071e76e07de34)


上分策略主要集中在数据清洗、推理速度优化和损失函数优化:

### 数据清洗
可以看出测试数据和训练数据的domain不一致，因此采用下采样两倍后的训练数据可以极大提升网络性能。同时不必要的空白图块也会干扰网络学习到去模糊的知识，让网络误以为恢复空白图块可以恢复的很好导致收敛不充分。因此采用了数据筛选后，我们的模型性能可以进一步提升。

### 推理速度

原始代码推理速度1.67s,耗时严重，经分析，我们采用更小的模型，在精度损失不大的情况下可以获得0.83s的速度， 网络满足线上V100推理显存需求

### 损失函数优化
一个很明显的情况是线上PSNR的指标很高，因此需要考虑如何提升ms_ssim指标。在网络第一阶段收敛后，结合Charbonnier Loss和MS_SSIM Loss进行分数性能的提升。

### Test-time Local Converter (TLC)优化
由于训练是在一个固定的patch size上进行训练的，这导致在处理大图块的过程中，不能充分利用所有信息，因此采用了TLC的模型优化策略来进一步提升网络模型。同时也导致了一定的时间损耗。

# 代码启动过程
根据main.ipynb可以直接运行代码