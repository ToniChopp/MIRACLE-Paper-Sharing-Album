# <center>YOLOX: Exceeding YOLO Series in 2021</center>
## <center>By pujunting</center>

## 1.摘要


&emsp;&emsp;YOLOX: 在2021年超过YOLO系列
&emsp;&emsp;[论文链接](https://arxiv.org/abs/2107.08430)
&emsp;&emsp;[代码链接](https://github.com/Megvii-BaseDetection/YOLOX)

&emsp;&emsp;在本文中介绍了对于YOLO系列的改进，形成了一种新的高性能检测器:YOLOX。具体方法包括有将YOLO检测器切换到anchor-free方式以及其他先进检测技术，例如decoupled head和领先的标签分配策略SimOTA，让模型在目前大物体检测中取得了最好的结果。

&emsp;&emsp;下面包括了一些文中提到的实验结果：对于YOLO-Nano仅用0.91M参数和1.08G FLOPs，在COCO上获得25.3%的AP，超过NanoDet 1.8%的AP；对于工业上应用最广泛的探测器之一YOLOv3，在COCO上将其提升至47.3%AP，比目前的最佳实践高出3.0%AP；对于参数量与YOLOv4CSP、YOLOv5-L大致相同的YOLOX-L，在COCO上实现了50.0%的AP，在特斯拉 V100上的速度为68.9FPS，超过YOLOv5-L1.8%的AP。此外，还凭借YOLOX-L模型赢得了流媒体感知挑战赛(2021 CVPR自动驾驶研讨会)的第一名。

<center>
<img src = "./src/fig1.PNG">
</center>

## 2. Introduction

&emsp;&emsp;随着目标检测的发展，YOLO系列始终追求实时应用的最佳速度和精度平衡。他们提取当时可用的最先进的检测技术(例如，针对YOLOv2的锚、针对YOLOv3的残差网络)，并针对最佳实践优化实施。目前，YOLOv5在13.7ms的COCO上以48.2%的AP保持了最佳的折衷性能。

&emsp;&emsp;然而，在过去的两年中，目标检测学术界的主要进展集中在无锚检测器，高级标签分配策略和端到端(无NMS)检测器。这些尚未融入YOLO家庭，如YOLOv4和YOLOv5仍然是基于锚的检测器，具有手工制作的训练分配规则。

&emsp;&emsp;因此，本文基于上述进展对于YOLO系列带来了优化与进步。

## 3. YOLOX

### 3.1 YOLOX-DarkNet53

#### 1) 实现细节

&emsp;&emsp;从基线到最终模型，训练设置基本一致。本文在COCO train2017上对模型进行了总共300个时期的训练，其中包括5个时期的预热，使用随机梯度下降进行训练。使用的学习率为lr×BatchSize/64(线性缩放)，初始化lr = 0.01，lr随着时间余弦变化。权重衰减为0.0005，SGD动量为0.9。
<center>
<img src = "./src/table1.PNG">
</center>

#### 2) YOLOv3 baseline

&emsp;&emsp;基线采用了DarkNet53主干和SPP层的架构，在一些论文中被称为yolov3-SPP。与最初的实现相比，本文稍微改变了一些训练策略，增加了均线权重更新、余弦lr调度、IoU损失和IoU感知分支。BCE损失用于训练cls和obj分支，IoU损失用于训练reg分支。

#### 3) Decoupled head

&emsp;&emsp;在目标检测中，分类和回归任务的冲突常常发生。因此，用于分类和定位的解耦头广泛用于大多数单级和两级检测器。然而，作为YOLO系列的主干和特征金字塔不断发展，它们的检测头保持耦合，如图所示。

&emsp;&emsp;文中所指的两个分析实验表明，耦合探测头可能会损害性能。1).用解耦的头代替YOLO的头大大提高了收敛速度。2).解耦的头部对于端到端版本的YOLO是必不可少的。端对端特性随着耦合头降低4.2%的AP，而对于解耦合头降低到0.8%的AP。因此，研究中用如图所示的lite去耦头代替YOLO检测头。具体来说，它包含一个1 × 1卷积层以减小通道尺寸，其后是两个平行分支，分别有两个3 × 3卷积层。

<center>
<img src = "./src/fig2.PNG">
</center>
<center>

&emsp;&emsp;下图是带有YOLOv3头或去耦头的检测器的训练曲线，每10个epochs在COCO val上评估一次AP。解耦磁头比YOLOv3磁头收敛得更快，最终获得更好的结果。

<img src = "./src/fig3.PNG">
</center>

#### 4) Strong data augmentation

&emsp;&emsp;研究中将Mosaic和MixUp添加到增强策略中，以提高YOLOX的性能。Mosaic是ultralytics-YOLOv3提出的一种有效的扩增策略。然后，它被广泛用于YOLOv4、YOLOv5和其他检测器。MixUp最初是为图像分类任务而设计的，但在对象检测训练中，MixUp的任务是很复杂的。本文在模型中采用了MixUp和Mosaic实现。

#### 5) Anchor-free

&emsp;&emsp;无锚机制显著减少了需要启发式调整的设计参数数量和涉及的许多技巧(例如，锚聚类，网格敏感)。它使得检测器，尤其是其训练和解码阶段变得相当简单。

&emsp;&emsp;将YOLO切换到无锚模式的做法具体是，将每个位置的预测从3减少到1，并使它们直接预测四个值，即网格左上角的两个偏移量，以及预测框的高度和宽度。将指派每个对象的中心位置作为正样本，并预先定义一个标度范围，以指定每个对象的FPN水平。这种修改减少了检测器的参数和GFLOPs，使其更快，但获得了更好的性能：42.9%的AP。

<img src = "./src/table2.PNG">
</center>

#### 6) Multi positives

&emsp;&emsp;为了与YOLOv3的分配规则保持一致，上述无锚点版本只为每个对象选择一个正样本(中心位置)，同时忽略其他高质量预测。然而，优化那些高质量的预测也可能带来有益的梯度，这可能缓解训练期间正/负采样的极端不平衡。本文简单地将中心3×3区域指定为正样本。

#### 7) SimOTA

&emsp;&emsp;本文总结了高级标签分配的四个关键点:1)损失/质量意识，2)居中在先，3)每个真实地面的正锚点的动态数量4(动态top-k)，4)全局视图。

&emsp;&emsp;SimOTA首先计算成对匹配度，由每个预测gt对的成本或质量表示。例如，在SimOTA中，$g_i$和预测$p_j$之间的成本计算如下:

$$c_\mathbf{i,j}=L_\mathbf{i,j}^\mathbf{cls}+\lambda*L_\mathbf{i,j}^\mathbf{reg}$$

&emsp;&emsp;其中λ是平衡系数。$L_{ij}^{cls}​​和L_{ij}^{reg}$​是$g_i$​和预测$p_j$​之间的分类损失和回归损失。然后，对于$g_i$​，选择固定中心区域内成本最小的前k个预测作为其正样本。最后，那些正预测的对应网格被指定为正，而其余网格为负。

&emsp;&emsp;SimOTA不仅减少了训练时间，而且避免了Sinkhorn-Knopp算法中额外的求解超参数。

#### 8) 端到端YOLO

&emsp;&emsp;本文添加两个额外的conv层，包括一对一的标签分配和停止梯度。这些使检测器能够以端到端的方式执行，但会略微降低性能和推理速度

### 3.2 其他骨干网络

&emsp;&emsp;除了DarkNet53，文章还在其他不同大小的主干上测试YOLOX，其中YOLOX相对于所有对应的主干实现了一致的改进。
<img src = "./src/table3.PNG">
</center>
<img src = "./src/table4.PNG">
</center>

&emsp;&emsp;为了进行公平的比较，文章采用了精确的YOLOv5主干，包括修改的CSPNet、SiLU激活和PAN头。

&emsp;&emsp;本文将模型进一步缩小为YOLOX-Tiny，用来与YOLOv4-Tiny进行比较。对于移动设备，采用深度方向卷积构建了一个YOLOX-Nano模型，该模型只有0.91M的参数和1.08G的FLOPs。

&emsp;&emsp;在实验中，所有的模型保持几乎相同的学习进度和优化参数.然而，作者发现不同尺寸的模型中，合适的增强策略是不同的。对YOLOX-Nano这样的小型模型来说，削弱增强效果更好。具体来说，在训练小模型即YOLOX-S、YOLOX-Tiny和YOLOX-Nano时，去掉了混叠增强，弱化了马赛克(将比例范围从[0.1，2.0]缩小到[0.5，1.5])。

&emsp;&emsp;对于大型模型，作者发现增强更强更有帮助。本文在混合图像之前，通过随机采样的比例因子对两幅图像进行抖动。

<img src = "./src/table5.PNG">
</center>

## 4.总结

&emsp;&emsp;在文章中介绍了基于YOLO系列的一些优化进步，形成了一个高性能的无锚探测器YOLOX。YOLOX配备了一些最新的检测技术，即去耦头、无锚和高级标签分配策略，在速度和精度之间取得了比其他同类产品更好的平衡。

&emsp;&emsp;个人感悟是，本文其实并未提出开创性的理论研究，但基于现有研究成果，敏锐地发现了当前解决方案的不足之处，并结合先进研究成果与YOLO系列，通过大量实验产生了新的方案，同样也是一种巨大的创新，应用性极强。