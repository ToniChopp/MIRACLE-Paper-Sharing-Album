# Miracle@USTC.CS Monthly Work Report(2021 Dec. 2022 Jan.)

> Our Work: Mainly rereading and intensive reading of the project team's paper sharing work, and making some notes.
>
> Our Github link:https://github.com/ToniChopp/MIRACLE-Paper-Sharing-Album

## 1.Unsupervised Deep Video Denoising

<p align="right">Excerpt By：Rongsheng Wang</p>

&emsp;&emsp;出自ICCV 2021。论文lib:https://arxiv.org/abs/2011.15045 &emsp;code:https://github.com/sreyas-mohan/udvd

&emsp;&emsp;本篇论文首先提出要解决的问题：当前视频去噪领域最为先进的方法是**使用CNN网络**，但需要使用干净的数据做训练。在诸如显微镜的应用中，无噪音的ground truth视频往往是不可用的。为了解决这个问题，本文提出了一种在没有监督数据的情况下训练视频去噪CNN的方法，称之为无监督的深度视频去噪（UDVD）。

&emsp;&emsp;随后，本文进行了多重实验，尽管没有干净的视频作为训练集，UDVD的性能与基于监督学习的模型相差不大。本文也验证了**当使用积极的数据增强和早期停止相结合时，即使只对单一的短暂的噪声视频序列（少至30帧）进行训练，它也能产生高质量的去噪效果**。最后，验证了UDVD在实际应用数据集中的可行性。

本文的贡献为：

- 设计了一种新的盲点架构/目标，用于无监督的视频去噪，实现了与最先进的监督方法相竞争的性能。
- 使用积极的数据增强（时间和空间反转）和早期停止的训练方法，通过对单一的短暂噪声视频的训练达到最先进的性能。
- 展示了提出的方法在对真实世界的电子和荧光显微镜数据以及原始视频去噪方面的有效性。与大多数现有的无监督视频去噪方法不同，该方法不需要预训练，这在真实世界的成像应用中是很关键的。
- 对UDVD学到的去噪机制进行分析，证明它可以进行隐性的运动补偿，尽管它只被训练为去噪。将该分析应用于监督网络，表明同样的结论是成立的。



## 2. Deformed2Self: Self-Supervised Denoising for Dynamic 

<p align="right">Excerpt By：Adler Xu</p>				

​		本文提出了一个用于动态成像去噪的深度学习框架，通过把不同时间帧的图像变形到目标帧来探索动态成像中图像的相似性，同时利用了“不同观测的噪声是独立的并且遵循相似的噪声模型”这一事实。

- 整个**pipeline**可以用**end-to-end**的方式进行训练，便于优化。
- 该模型是自监督的，不需要**clean-vs-noisy image pairs**，只需要噪声图像就可以训练。
- 该模型可以在**single image**上训练 **(with a few auxiliary observations)**，并且不需要大的训练数据集，适用于数据稀缺的应用场景。
- 通过**sequential single- and multi-image denoising networks**来优化 **image quality**，在各种噪声设置的实验中，该模型与其他最先进的无监督或自监督去噪方法具有相当甚至更好的性能。


## 3. Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels

<p align="right">Excerpt By：Zhen Huang</p>

​		出自NIPS2018，论文lib：https://papers.nips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networks-with-noisy-labels.pdf。

​		几乎所有的数据集都因为不可抗力原因含有错误标注数据，而有噪数据和无噪数据以及噪声比例的不同，都会影响到实验结果。所以**带噪学习领域**的研究是十分有意义的。

​		针对有噪数据，文章提出了一种**结合了MAE和CCE**的新损失函数（GCE），充分利用了MAE的对称性带来的可以有效抑制噪声数据，以及其存在**收敛速度慢、训练困难**的问题和CCE的无对称、快速收敛性质。针对不同的数据集（cifar10，cifar100，fashion-mnist等）设置不同的噪声比例，最终都取得了SOTA结果。作者认为，该损失函数可以很容易地应用于任何现有的DNN架构和算法，同时在各种有噪声的标签场景中产生良好的性能。

​		同时作者还提出了一种剪枝策略，当预测值p>k时才进行训练，否则认为是噪声直接不参与训练。这样的改进在大多数测试的噪音比例的情况下都取得了更优的结果。

​		尚未解决的问题：该paper假设标注错误的概率为p，实际情况是因为类别不同可能带来不同的p，此时**对称性就不再成立**。同时，组里讨论的时候也认为**漏标和错标情况**并不完全相同，也需要进一步讨论做对比试验。

​		本文启发较多，给了不少trick。在去除/纠正噪声数据、设计改进网络较为费时的情况下，改进loss function也不失为应对有噪数据的好方法。这种方法同样可以迁移到**数据均衡、数据增强**等领域。



## 4. DT-MIL: Deformable Transformer for Multi-instance Learning on Histopathological Image

<p align="right">Excerpt By：Huijie Hu</p>

&emsp;&emsp;出自MICCAI 2021。论文lib:https://rdcu.be/cyl9Z  ​&emsp;&emsp;code:https://github.com/yfzon/DT-MIL.

&emsp;&emsp;由于成像技术的发展，通常病理图像的尺寸巨大，可以将其视为很多的实例的拼接。但是在进行图像的分析的时候，由于要考虑到其中的各类微环境，需将图像综合来分析，那么这就是一个Multi-instance Learning（MIL）任务。嵌入空间的方法（ES-MIL）能够较为理想地进行全局影像分析。本文创新地将transformer引入到医学影像的MIL中，提出了一种ES-MIL的模型：DT-MIL，它包含三个模块：保持位置的降维（PPDR）、基于transformer的包嵌入（TBBE）和分类器。

本论文通过实验可见注意力机制用于大图像的分类和预测任务的巨大优越性。而对于不同领域中优秀模型的迁移，实际上要求精心去设计相关的流程与细节。我认为本文在这方面的工作比较有价值，同时设计出多种基于Transformer的ES-MIL方法，并采取其中表现最优的模型，创新点明显。

## 5. Targeted Gradient Descent: A Novel Method for Convolutional Neural Networks Fine-tuning and Online-learning  

<p align="right">Excerpt By：Hao Zhang</p>

&emsp;&emsp;出自MICCAI-2021。论文lib：https://arxiv.org/abs/2103.13557  &emsp;&emsp;code：https://github.com/DIAL-RPI/TASK-Oriented-CT-Denoising_TOD-Net

&emsp;&emsp;医用CT的广泛应用引起公众对CT辐射量的关注，降低辐射剂量会增加CT成像的噪声和伪影，影响了医生的判断和下游医学图像分析任务的性能。近年基于深度学习的方法去噪得到的一定应用，但现有的方法都是与下游任务无关的。本文引入了一种新的面向任务的去噪网络(TOD-Net)，该网络利用了来自下游任务的知识来实现面向任务的损失。通过系列实证分析表明，任务导向损失弥补了其他任务无关损失，通过控制去噪器来提高任务相关区域的图像质量。这种增强反过来又为下游任务带来了各种方法的性能的普遍提高。
&emsp;&emsp;个人认为本文针对现有深度学习方法图像去噪中忽视下游任务问题上，在WGAN框架基础上，提出了结合下游任务同时训练的去噪模型，建立了Task-oriented Loss，十分具有启发性。该方法同样可以迁移至很多其他上游任务处理，而不是仅仅局限于图像去噪上。

## 6. DeepDRR -- A Catalyst for Machine Learning in Fluoroscopy-guided Procedures

<p align="right">Excerpt By：Hao Zhang</p>

&emsp;&emsp;出自MICCAI 2018，论文lib：https://arxiv.org/pdf/1803.08606
&emsp;&emsp;在与诊断放射学相关的大多数学科中，基于机器学习方法优于竞争方法。但目前介入放射学尚未从深度学习中受益，主要有两个原因：第一手术过程中获得的大多数图像从未保存；其次即使保存了图像，由于大量的数据，相关的注释是个挑战。在考虑透视引导程序中，真正介入透视检查的替代方案是通过3D诊断CT对手术进行计算机模拟。这样标记相对容易获得，但是生成的合成数据正确性取决于前向模型，据此本文提出DeepDRR，这是一个用于从CT扫描中快速、真实地模拟透视和数字放射成像的框架，与深度学习的原生软件平台紧密结合。本文分别使用机器学习在 3D 和 2D 中进行Material Decomposition和散点估计，并结合解析前向投影和噪声注入来实现所需的性能。在骨盆 X 射线图像中的解剖标志检测示例中，本文证明了在DeepDRR上训练的机器学习模型可以推广到临床采集数据，而无需重新训练或领域适应。
&emsp;&emsp;**DeepDRR**： 本文提出的基于Python、PyCUDA和PyTorch的框架，用于快速自动模拟CT数据中的X射线图像，由4个模块组成：1）使用深度分割ConvNet对CT体积进行Material Decomposition；2）一物质和光谱感知的光线追踪正向投影仪；3）基于神经网络的Rayleigh scatter estimation；4）噪声注入。


## 7. Anchor-guided online meta adaptation for fast one-Shot instrument segmentation from robot

<p align="right">Excerpt By：Zhen Huang</p>

​		论文lib：https://doi.org/10.1016/j.media.2021.102240。 		

​		机器人辅助手术(RAS)中带注释的手术数据的缺乏，促使以往的研究借鉴相关领域知识，通过适应性的方法对手术图像实现有前景的分割结果。本文的贡献在于提出了anchor-guided online meta adaptation (AOMA)。

​		具体的：通过元学习实现了快速的一次测试时间优化，从源视频中获得了良好的模型初始化和学习率，避免了费力的手工微调。在具有匹配感知损失的特定视频任务空间中优化可训练的两个组件。此外，设计了anchor-guided online adaptation，来解决整个机器人手术序列的性能下降。该模型在anchor matching支持的motion-insensitive pseudo-masks上都能很好适用。AOMA在两种实际场景下取得了最先进的结果:(1)普通视频到手术视频，(2)公开手术视频到内部手术视频，同时大大减少了测试运行时间。

主要贡献在于：

1.解决了机器人手术视频的一次性仪器分割问题，只需每个视频的第一帧掩模即可快速适配。

2.设计一种anchor-guided online adaptation，连续地对anchor maching生成的第一帧掩码和随后的pseudo scalar进行adaptation。对于运动信号不敏感，可以很好地解决机器人手术视频中存在的快速器械运动问题。

3.提出通过匹配感知优化过程元学习最优模型初始化和学习速度，实现快速online adaptation。

**Expectation：**1.进行online adaptation时不对整个模型更新，只对一些层更新，收敛的更快。

2.对部分效果不好的pseudo scalar（noise）采用过滤/剪枝等方法。



## 8. nnFormer Interleaved Transformer for Volumetric Segmentation

<p align="right">Excerpt By:  Jiakun Wang</p>

&emsp;&emsp;论文链接：https://arxiv.org/abs/2109.03201

&emsp;&emsp;**主要工作：**
&emsp;&emsp;在本文中，作者提出了一种名为nnFormer的新的医学图像分割网络。nnFormer是在卷积和自我注意力机制的交错结构上构建的，卷积结构有助于将精准的空间信息编码为高分辨率的低层次特征，并在多个尺度上建立层次化的概念。另一方面，Transformer中的自我注意力机制将长距离依赖与卷积表征纠缠在一起，以捕捉全局背景。在这种混合架构中，nnFormer比以前基于Transformer的分割方法取得了巨大的进步。

&emsp;&emsp;**个人总结：**
&emsp;&emsp;本文作者提出的这种网络结构借鉴了uNet,只是把卷积块替换成了Swin transformer块，利用分层的方式进行self-attention即在三维空间内计算而不是传统的二维，作者把这种叫做V-MSA。利用交错式的结构建模。和uNet一样，nnformer的encoder部分是下采样的，decoder部分是上采样的。阅读代码发现，代码框架也使用了nnUnet的框架，包括预处理，后处理，网络结构和训练推理部分。

&emsp;&emsp;**启发：**
&emsp;&emsp;其实往小里说，nnFormer不过是基于Swin Transformer和nnUNet的经验结合，technical上的novelty并不多。但是往大里说的话，nnFormer其实是一个很好的起点，可以启发更多的人投入到相关的topic中开发出更好的基于Transformer的医疗影像分析模型。

## 9. Distilling Knowledge via Knowledge Review

<p align="right">Excerpt By：Rongsheng Wang</p>

&emsp;&emsp;出自CVPR 2021。论文lib: https://arxiv.org/abs/2104.09044 &emsp;code: https://github.com/dvlab-research/ReviewKD

&emsp;&emsp;本文从以下问题出发：深度卷积神经网络（CNN）已经在各种计算机视觉任务中取得了显著的成功，但CNN的成功往往伴随着相当大的计算量和内存消耗，这使得它在资源有限的设备上的应用成为一个具有挑战性的课题。

&emsp;&emsp;本文研究了以前被忽视的在知识蒸馏中设计连接路径的重要性，并相应提出了一个新的有效框架。关键的修改是利用教师网络中的低级特征来监督学生的深层特征，从而使整体性能得到很大的提高。同时进一步分析网络结构，发现学生的高层次阶段有很大能力从教师的低层次特征中学习有用的信息。类似人类学习过程，本文使用了一个知识回顾框架，使用教师的多层次信息来指导学生网络的单层次学习。

本文的主要贡献为：

- 在知识蒸馏中提出了一种新的审查机制，利用教师的多层次信息来指导学生网的单层次学习。
- 提出了一个剩余学习框架，以更好地实现审查机制的学习过程。
- 为了进一步改进知识审查机制，提出了一个基于注意力的融合（ABF）模块和一个层次化的上下文损失（HCL）函数。
- 通过应用本文的蒸馏框架，在多个计算机视觉任务中实现了许多紧凑模型的最先进性能。

&emsp;&emsp;个人认为本文的不足在于：金字塔池化可能会有问题，student和teacher变成多级可能会更好？

## 13.Fair Attribute Classification through Latent Space De-biasing

<p align="right">Excerpt By:  Huijie Hu</p>

&emsp;&emsp;出自CVPR 2021。论文lib:https://arxiv.org/abs/2012.01469  ​&emsp;&emsp;code:https://github.com/princetonvisualai/gan-debiasing

&emsp;&emsp;在计算机视觉领域的任务中，视觉识别的公平性非常关键。当使用 GAN 生成图像，由于图像中的属性本身可能具有关联性，会影响生成图像的公平性，尤其是在数据集中数据不充分的情况下，本文提出了一种平衡属性的公平性的方法。将属性分为受保护属性和目标属性，将其在线性的隐空间内抽象，找到其相应的超平面。那么对于一个生成的扰动$z$，就可以找到一个举例目标属性超平面距离相等，但距离受保护属性超平面距离不定的点$z'$，用于生成一系列的图像，做数据增强。

&emsp;&emsp;本文的贡献在于引入了一种基于 GAN 的数据增强方法来训练更公平的属性分类器，对于在各种环境中增强 GAN 潜在空间中的数据有广阔的应用前景。本文的创新点在于从隐空间去消除数据的不公平性，其中的数学推导和在模型训练中的细节处理方法很具有价值。

## 14. Few-Shot Domain Adaptation with Polymorphic Transformers

<p align="right">Excerpt By:  Yuandong Liu</p>

论文出自 MICCAI 2021，论文链接：https://arxiv.org/abs/2107.04805

一般 few shot learning 需要或假设目标域和源域样本来自同一域，或者它们数据空间尽可能对齐，但往往不是这样。本文提出了一种方法解决 DA 的问题，使用一个改进的 transformer 模块。使用时将该模块嵌入到一个已有预训练好模型的 feature extracter 和 task head 之间。这个新模型在 source domain 上 fine-tuning，然后在 target domain 上训练。当在目标域上训练时，这个模块的输入是 source domain 图像的 feature 和 target domain 的 prototype。这样做的目的是，通过 transformer 的 attention 来实现 source domain 和 target domain 的对齐。在 target domain 上只需要更新这一个模块。作者通过在眼底盘分割和息肉分割两个任务上的实验证明该方法的有效性。

## 15.Segmenter：Transformer for Semantic Segmentation

<p align="right">Excerpt By:  Adler Xu</p>

#### 贡献

​		本文有四个贡献

- 基于**Vision Transformer (ViT)**对语义分割提出了一个方法：它不使用卷积，经过设计后能捕捉**contextual information**，并且性能超过了**FCN-based**的方法。
- 提出了一系列具有不同分辨率级别的模型，可以在精确度和运行时间之间进行**trade-off**。
- 提出了一个生成**class masks**的decoder，性能更胜一筹，并且能够扩展以执行更**general**的图像分割任务。
- 提出的方法在**ADE20K**和**Pascal Context**数据集上展现出了最好的结果，

#### 结论

​		提出了一个用于语义分割的transformer模型，名为**Segmenter**。

- 与最近的**Vision Transformer (ViT)**不同的是，所有的**images patches**都在encoding中发挥了作用，并且这个模型能够很好的捕捉到**global context**的信息。
- 对**patch encodings**运用**simple point-wise linear decoder**到达了很好的性能，并且用**mask transformer**解码时，性能近一步提升。

## 16. A Convnet for the 2020s

<p align="right">Excerpt By:  Jiakun Wang</p>

&emsp;&emsp;论文链接：https://arxiv.org/abs/2201.03545
&emsp;&emsp;**主要工作：**
&emsp;&emsp;原Facebook AI Research的研究，该研究梳理了从ResNet到类似于Transformer的卷积神经网络的发展轨迹，为了研究Swin Transformer的设计和标准卷积神经网络的简单性，从ResNet20出发，首先使用用于训练视觉Transformer的训练方法对其进行训练，与原始ResNet50相比性能获得了很大的提升，并将改进后的结果作为基线。该研究制定了一系列设计决策，总结为1宏观设计 2ResNeXt 3反转瓶颈 4卷积核大小 5各种逐层微设计，所有模型均在Imag1k上进行训练和评估，并粗略控制了FLOPs。
&emsp;&emsp;研究者提出了一个叫做ConvNetx的纯ConvNets系列并在各种任务上进行评估，ConvNets在准确性和可扩展性方面取得了与Transformer具有竞争力的结果，同时保持标准ConvNet的简单性和有效性。


&emsp;&emsp;**个人总结：**
&emsp;&emsp;堆了很多trick,包括优化器如AdamW,数据增强如mixup，随机深度和标签平滑，以及GELU,LN以及分离式下采样。还有一个最重要的点是使用了更大的kernel７ｘ７而不是传统的３ｘ３，作者在这里解释说transformer的特性是非局部注意力机制获得全局感受野，虽然swin采用了局部窗口，但也是７ｘ７，所以对照swin使用了更大的卷积核。最后实验结果在imagenet和下游任务评估上取得了和swin差不多的结果或者略有超过。

&emsp;&emsp;**启发：**
&emsp;&emsp;Fair的这篇文章感觉更多的是堆trick，做了非常多的实验也足够严谨。个人感觉像Swin这类的分层Transformer越来越接近cnn的设计，如分层下采样和滑动窗口等但又不如cnn优美自然，反观ViT这种原汁原味的attention机制，没有使用任何的先验信息，这是否有一种绕回去的感觉。还有的疑惑就是，如果参数量足够大的话，模型结构是否已经不重要了，因为已经足够拟合，从ConvNeXt和swin表现出的性能相当来看，应该更多的从数据和训练方法等方面研究和改进？个人觉得transformer在dl上有更好的物理解释性但transformer在图像操作上应该还是有信息冗余，每一个patch都做attention对信息提取有用但不一定高效率，应该要设计一些辅助网络帮助更好的学习局部特征减少信息冗余，这对transformer在部署端也会有更大的改进，图像是局部信息为主的模态，而文本则是全局信息为主，在新的算子提出来之前，应该是attention和cnn的继续结合吧。


## 17. UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss

<p align="right">Excerpt By:  Hao Zhang</p>

&emsp;&emsp;出自AAAI 2018，code：https://github.com/tornadomeet/UnFlow ，论文lib：https://arxiv.org/pdf/1711.07837  
&emsp;&emsp;最近用于光流的端到端卷积网络依赖于合成数据集进行监督，但训练和测试场景之间的域不匹配是一个挑战。受经典的基于能量的光流方法的启发，本文设计了一种基于遮挡感知双向流估计和robust census transform的无监督损失，以克服对实际流的需求。在 KITTI 基准测试中，我们的无监督方法大大优于以前的无监督深度网络，甚至比仅在合成数据集上训练的类似监督方法更准确。通过选择性地对 KITTI 训练数据进行微调，我们的方法在 KITTI 2012 和 2015 基准上实现了具有竞争力的光流精度，因此此外还可以对具有有限真实数据的数据集的监督网络进行通用预训练。
&emsp;&emsp;个人初始感觉估计遮挡部分会很难收敛，但实际效果还可以。后续的无监督论文也挺多遵照这个思路去估算遮挡部分。