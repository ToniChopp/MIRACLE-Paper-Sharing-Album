# Miracle@USTC.CS Monthly Work Report(Nov.)

> Our Work: Mainly rereading and intensive reading of the project team's paper sharing work, and making some notes.
>
> Our Github link:https://github.com/ToniChopp/MIRACLE-Paper-Sharing-Album

## 1.Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation

​<p align="right">Excerpt By：Zhen Huang</p>

​&emsp;&emsp;出自MICCAI 2021 oral。论文lib:https://arxiv.org/abs/2106.13292 &emsp;code:https://github.com/vios-s/DGNet

&emsp;&emsp;本篇论文重点聚焦于Domain generalisation（DG，领域泛化）和Disentanglement（解耦），以及Meta-learning（元学习），并应用在医学图像分割场景中。文中提到的概念都很新颖，提出了一个**半监督元学习解耦模型**来解决问题：首先对domain shifts的相关表征进行建模，通过解耦这些表示并重组来重建输入图像，使得可以使用**未标记数据**来更好地贴合元学习的实际域迁移从而获得更好的泛化性能，尤其是标记样本有限的情况下。最终在当前的公开数据集M&M以及SCGM中都取得了比baseline更优的结果。使用了low-rank的正则化作为学习的bias来提高解耦从而提升泛化性能也是文章的一大亮点。

&emsp;&emsp;当前迁移学习还存在困难--将模型推广到新domain上的新数据时泛化能力往往达不到，从而最终在未知的数据集（unseen）上也达不到非常理想的效果。这很大程度是因为源数据和未知域的数据上存在domain shifts。对于此问题，可以借鉴本文的做法使用基于梯度的元学习策略（gradient-based meta-learning）。**其他域**的补充**未标记数据**能否进一步提升模型性能是未来工作的一个方向。


## 2.Orthogonal Ensemble Networks for Biomedical Image Segmentation

<p align="right">Excerpt By: Rongsheng Wang</p>

&emsp;&emsp;出自MICCAI 2021 pp 594-603。论文lib:https://arxiv.org/abs/2105.10827 &emsp;code:https://github.com/agosl/Orthogonal_Ensemble_Networks

&emsp;&emsp;本篇论文首先概述了一个先前的Deep Learning模型的缺点：**校准能力差，导致预测结果过于保守**。随后提出使用**集合学习**这一学习策略，可以提高模型的鲁棒性和校准性能，通过**正交约束**可以诱导模型多样性，使得在图像分割领域模型有更好的表现。接着在*BraTS 2020*数据集上验证了本文提出的观点。本文贡献为：提出了一个可以**提高深度卷积神经网络(DCNN)集合的模型多样性**的策略，即假设可以利用集合的模型的正交化来提升模型整体的多样性，从而提高预测性能和更好的校准模型输出。

&emsp;&emsp;本篇论文提出的使用正交约束来把模型集合化是一个很新颖的策略，对图像分割工作提供了思路。同时对构建新的模型架构有一个启发：利用正交性的数学性质，增加模型的多样性，提升校准性能。


## 3. Pose2Mesh_ Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose

​<p align="right">Excerpt By：Huijie Hu</p>

&emsp;&emsp;出自ECCV 2020.论文lib: http://arxiv.org/abs/2008.09047   &emsp;code: https://github.com/hongsukchoi/Pose2Mesh_RELEASE

&emsp;&emsp;本篇论文针对人体三维pose和mesh估计中常见的两大问题（训练的模型在有复杂环境的图片中的预测生成不能表现很好；由于三维旋转的存在pose参数不能很好应用于回归），创新性提出了一种叫Pose2Mesh的模型，该模型用到了图卷积，直接由回归生成mesh的节点坐标数据。Pose2Mesh具有级联结构，整体的流程是先由二维的pose变为三维的pose。在MeshNet中，把二维和三维的pose作为输入，然后通过持续的上采样，由粗到细构建三维的mesh。

&emsp;&emsp;本文采用**最近邻插值法**进行上采样，实现模型的coarse-to-fine。实验在**Human3.6M**、**3DPM**等数据集上训练和测试，均取得了很好的性能，说明了图卷积和级联结构用在从图片构建三维mesh模型这一工作上的优越性。



## 4.Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video
<p align="right">Excerpt By：Wang Jiakun</p>


&emsp;&emsp;出自MICCAI 2021.论文lib: https://arxiv.org/abs/2109.13593  &emsp;code：https://github.com/jcwang123/DMNet

&emsp;&emsp;本文针对机器人手术视频的器械分割提出了一种名为Dual-Memory Net的网络结构，主要思想是在视频的处理上不仅要考虑当前帧，还要纳入时间维度上的其他帧信号。网络的结构包括两部分，一种是局部的相邻帧进行聚合，称为ELA模块，ELA模块输出特征增强后导入一个叫做AGA的模块，此模块会主动纳入从视频开始到现在最具信息量和最有代表性的帧，最后一起聚合再进行特征增强，可以提高器械分割的特征提取效率和鲁棒性。网络的特征提取用了ConvLSTM和Non-Local机制增加感受野。

&emsp;&emsp;目前在视频的器械分割上提出的其他方法要么没有考虑相邻帧可能携带的信息，要么会带来巨大的算力成本。作者提出的这种结构充分考虑了视频流的时间连续性，在视频的处理上可以利用相邻帧进行特征增强.存在的疑惑：实时视频的分割也要考虑时间成本，本文并未提到这一点，使用全局注意力机制和不断的存储帧是否会增加计算时间，需要实验验证。

## 5.Targeted Gradient Descent: A Novel Method for Convolutional Neural Networks Fine-tuning and Online-learning
<p align="right">Excerpt By：Pu junting</p>


&emsp;&emsp;出自MICCAI 2021.论文lib: https://arxiv.org/abs/2109.14729  &emsp;code：none

&emsp;&emsp;卷积神经网络(ConvNet)通常使用来自相同分布绘制的图像进行训练和测试。要将ConvNet推广到各种任务，通常需要一个完整的训练数据集，其中包含从不同任务中提取的图像。在大多数情况下，几乎不可能预先收集所有可能的代表性数据集。只有在临床实践中部署ConvNet后，新数据才可用。然而，ConvNet可能会在分布不均匀的测试样本上产生伪像。

&emsp;&emsp;本文介绍了目标梯度下降，这是一种新的增量学习方案，可在已训练网络中有效地重用冗余内核。所提出的方法可以很容易地作为一个层插入到现有网络中，并且不需要重新访问先前任务中的数据。更重要的是，它可以实现测试研究的在线学习，以增强网络在实际应用中的泛化能力。