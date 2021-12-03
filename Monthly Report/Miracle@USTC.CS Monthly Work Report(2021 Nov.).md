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