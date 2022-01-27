# Miracle@USTC.CS Monthly Work Report(2021 Dec. 2022 Jan.)

> Our Work: Mainly rereading and intensive reading of the project team's paper sharing work, and making some notes.
>
> Our Github link:https://github.com/ToniChopp/MIRACLE-Paper-Sharing-Album

## 1.Unsupervised Deep Video Denoising

​<p align="right">Excerpt By：Rongsheng Wang</p>

​&emsp;&emsp;出自ICCV 2021。论文lib:https://arxiv.org/abs/2011.15045 &emsp;code:https://github.com/sreyas-mohan/udvd

&emsp;&emsp;本篇论文首先提出要解决的问题：当前视频去噪领域最为先进的方法是**使用CNN网络**，但需要使用干净的数据做训练。在诸如显微镜的应用中，无噪音的ground truth视频往往是不可用的。为了解决这个问题，本文提出了一种在没有监督数据的情况下训练视频去噪CNN的方法，称之为无监督的深度视频去噪（UDVD）。

&emsp;&emsp;随后，本文进行了多重实验，尽管没有干净的视频作为训练集，UDVD的性能与基于监督学习的模型相差不大。本文也验证了**当使用积极的数据增强和早期停止相结合时，即使只对单一的短暂的噪声视频序列（少至30帧）进行训练，它也能产生高质量的去噪效果**。最后，验证了UDVD在实际应用数据集中的可行性。

本文的贡献为：

- 设计了一种新的盲点架构/目标，用于无监督的视频去噪，实现了与最先进的监督方法相竞争的性能。
- 使用积极的数据增强（时间和空间反转）和早期停止的训练方法，通过对单一的短暂噪声视频的训练达到最先进的性能。
- 展示了提出的方法在对真实世界的电子和荧光显微镜数据以及原始视频去噪方面的有效性。与大多数现有的无监督视频去噪方法不同，该方法不需要预训练，这在真实世界的成像应用中是很关键的。
- 对UDVD学到的去噪机制进行分析，证明它可以进行隐性的运动补偿，尽管它只被训练为去噪。将该分析应用于监督网络，表明同样的结论是成立的。


## 2. TBC

## 3. TBC

## 4. TBC

## 5. TBC

## 6. TBC

## 7. TBC

## 8. nnFormer Interleaved Transformer for Volumetric Segmentation
<p align="right">Excerpt By:  Jiakun Wang</p>
&emsp;&emsp;论文链接：https://arxiv.org/abs/2109.03201

&emsp;&emsp;主要工作：
&emsp;&emsp;在本文中，作者提出了一种名为nnFormer的新的医学图像分割网络。nnFormer是在卷积和自我注意力机制的交错结构上构建的，卷积结构有助于将精准的空间信息编码为高分辨率的低层次特征，并在多个尺度上建立层次化的概念。另一方面，Transformer中的自我注意力机制将长距离依赖与卷积表征纠缠在一起，以捕捉全局背景。在这种混合架构中，nnFormer比以前基于Transformer的分割方法取得了巨大的进步。

&emsp;&emsp;个人总结：
&emsp;&emsp;本文作者提出的这种网络结构借鉴了uNet,只是把卷积块替换成了Swin transformer块，利用分层的方式进行self-attention即在三维空间内计算而不是传统的二维，作者把这种叫做V-MSA。利用交错式的结构建模。和uNet一样，nnformer的encoder部分是下采样的，decoder部分是上采样的。阅读代码发现，代码框架也使用了nnUnet的框架，包括预处理，后处理，网络结构和训练推理部分。

&emsp;&emsp;启发：
&emsp;&emsp;其实往小里说，nnFormer不过是基于Swin Transformer和nnUNet的经验结合，technical上的novelty并不多。但是往大里说的话，nnFormer其实是一个很好的起点，可以启发更多的人投入到相关的topic中开发出更好的基于Transformer的医疗影像分析模型。

## 9. Distilling Knowledge via Knowledge Review

<p align="right">Excerpt By：Rongsheng Wang</p>

​&emsp;&emsp;出自CVPR 2021。论文lib: https://arxiv.org/abs/2104.09044 &emsp;code: https://github.com/dvlab-research/ReviewKD

&emsp;&emsp;本文从以下问题出发：深度卷积神经网络（CNN）已经在各种计算机视觉任务中取得了显著的成功，但CNN的成功往往伴随着相当大的计算量和内存消耗，这使得它在资源有限的设备上的应用成为一个具有挑战性的课题。

&emsp;&emsp;本文研究了以前被忽视的在知识蒸馏中设计连接路径的重要性，并相应提出了一个新的有效框架。关键的修改是利用教师网络中的低级特征来监督学生的深层特征，从而使整体性能得到很大的提高。同时进一步分析网络结构，发现学生的高层次阶段有很大能力从教师的低层次特征中学习有用的信息。类似人类学习过程，本文使用了一个知识回顾框架，使用教师的多层次信息来指导学生网络的单层次学习。

本文的主要贡献为：

- 在知识蒸馏中提出了一种新的审查机制，利用教师的多层次信息来指导学生网的单层次学习。
- 提出了一个剩余学习框架，以更好地实现审查机制的学习过程。
- 为了进一步改进知识审查机制，提出了一个基于注意力的融合（ABF）模块和一个层次化的上下文损失（HCL）函数。
- 通过应用本文的蒸馏框架，在多个计算机视觉任务中实现了许多紧凑模型的最先进性能。

&emsp;&emsp;个人认为本文的不足在于：金字塔池化可能会有问题，student和teacher变成多级可能会更好？

## 16. A Convnet for the 2020s

<p align="right">Excerpt By:  Jiakun Wang</p>

&emsp;&emsp;论文链接：https://arxiv.org/abs/2201.03545
&emsp;&emsp;主要工作：
&emsp;&emsp;原Facebook AI Research的研究，该研究梳理了从ResNet到类似于Transformer的卷积神经网络的发展轨迹，为了研究Swin Transformer的设计和标准卷积神经网络的简单性，从ResNet20出发，首先使用用于训练视觉Transformer的训练方法对其进行训练，与原始ResNet50相比性能获得了很大的提升，并将改进后的结果作为基线。该研究制定了一系列设计决策，总结为1宏观设计 2ResNeXt 3反转瓶颈 4卷积核大小 5各种逐层微设计，所有模型均在Imag1k上进行训练和评估，并粗略控制了FLOPs。
&emsp;&emsp;研究者提出了一个叫做ConvNetx的纯ConvNets系列并在各种任务上进行评估，ConvNets在准确性和可扩展性方面取得了与Transformer具有竞争力的结果，同时保持标准ConvNet的简单性和有效性。


&emsp;&emsp;个人总结：
&emsp;&emsp;堆了很多trick,包括优化器如AdamW,数据增强如mixup，随机深度和标签平滑，以及GELU,LN以及分离式下采样。还有一个最重要的点是使用了更大的kernel７ｘ７而不是传统的３ｘ３，作者在这里解释说transformer的特性是非局部注意力机制获得全局感受野，虽然swin采用了局部窗口，但也是７ｘ７，所以对照swin使用了更大的卷积核。最后实验结果在imagenet和下游任务评估上取得了和swin差不多的结果或者略有超过。


&emsp;&emsp;启发：
&emsp;&emsp;Fair的这篇文章感觉更多的是堆trick，做了非常多的实验也足够严谨。个人感觉像Swin这类的分层Transformer越来越接近cnn的设计，如分层下采样和滑动窗口等但又不如cnn优美自然，反观ViT这种原汁原味的attention机制，没有使用任何的先验信息，这是否有一种绕回去的感觉。还有的疑惑就是，如果参数量足够大的话，模型结构是否已经不重要了，因为已经足够拟合，从ConvNeXt和swin表现出的性能相当来看，应该更多的从数据和训练方法等方面研究和改进？个人觉得transformer在dl上有更好的物理解释性但transformer在图像操作上应该还是有信息冗余，每一个patch都做attention对信息提取有用但不一定高效率，应该要设计一些辅助网络帮助更好的学习局部特征减少信息冗余，这对transformer在部署端也会有更大的改进，图像是局部信息为主的模态，而文本则是全局信息为主，在新的算子提出来之前，应该是attention和cnn的继续结合吧。
