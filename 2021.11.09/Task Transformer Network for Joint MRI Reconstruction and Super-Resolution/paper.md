## abstract

1. mri 的一个核心问题是速度和质量二者不可兼得，目前的方法的设计都是分开考虑，不考虑二者联系；
2. 超分辨率和 mri 重建是 mri 的两个重要研究手段或者说方向；
3. 这篇文章提出了端到端的 task transformer network ($\text{T}^\text{2}\text{Net}$)，联合两个任务重建和超分辨率，让特征在两个任务中共享，目的来实现高质量、超分辨率的、无运动伪影的图片；
4. 整个网络设计结构：先用 CNN 提取 task-specific features, 然后通过一个 task transformer 模块来联系两个分支；

mr 扫描的数据填充到 k 空间，从 k 空间采样得到数据，然后通过反傅里叶变换得到 mri 图像。

图像重建 image reconstruction和图像超分辨率是MRI的两个主要技术。前者通过减少 k 空间采样率来加速 MRI。后者通过恢复单个退化的低分辨率 (LR) 图像获得高分辨率 (HR) 图像.（不太懂什么意思。）

现有方法分别执行这两个任务，忽略了它们之间的关系。

这篇文章提出Task transformer network，共同执行图像重建和图像超分辨率，允许多个任务之间共享表示和特征变换。利用一个任务的知识加速另一个任务的学习过程。

## introduction


1. 重建和超分辨率已经有很多优秀的方法；比如[感知压缩](https://www.zhihu.com/question/28552876)、低秩约束、[字典学习](https://www.cnblogs.com/endlesscoding/p/10090866.html）、manifold fitting 等技术被运用到 mri 重建，其利用先验知识来克服因违背香农采样而导致的混叠伪影。
2. 深度神经网络被广泛应用，不同 CNN 方法被用来进行快速 mri 重建；典型例子：model-based unrolling methods，如$\text{VN-Net}$, $\text{ADMN-Net}$；端到端方法，如基于 U-Net, GAN。
3. 对于超分辨率，一些迭代算法如 low rank 或 sparese representation（稀疏表示），将 image priors（deep image priors？）考虑为正则项，然后通过低分辨率图片获取更高质量图片。类似的，CNN 方法也在超分辨率任务上实现了sota的结果，比如残差学习提取多尺度信息，或者基于 GAN 从低分辨率输入获得高分辨率输出。
4. multi-task learning 在自然语言处理、语音识别、计算机视觉等领域都取得了一定的成功

基于 transformer 和 multi-task learning，本文提出$\text{T}^\text{2}\text{Net}$，结合重建和超分辨率两个任务。

本文贡献有三：
1. 第一个把 transoformer 引入到重建和超分辨率的 multi-task learning 中；
2. 网络结构，下文将具体阐述
3. 相比各种 sota 的 重建和超分辨率的 sequential combination 模型（？），本文模型取得更好的结果。


## model

<img src="./image/framework%20overview.png">

## reference

1.https://www.jianshu.com/p/f102f4b23b90