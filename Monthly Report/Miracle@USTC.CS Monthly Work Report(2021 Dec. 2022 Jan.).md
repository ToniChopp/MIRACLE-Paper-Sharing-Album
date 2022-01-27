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

## 8. TBC

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