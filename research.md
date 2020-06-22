

## [DeepSpeed](https://github.com/microsoft/DeepSpeed)
---
### 几个核心问题：

> 评估计算力量【flops】
>  *  [pytorch 引入openai 计算量评估方法](https://discuss.pytorch.org/t/anyone-has-a-code-for-flop-calculation-of-an-epoch/49666/4)
>  *  [OpenAI 如何评估计算量](https://openai.com/blog/ai-and-compute/#fn2)


> 关于分布式：
> * DDP:  [blog 基础](https://leimao.github.io/blog/PyTorch-Distributed-Training/) 
> * DDP:  [官方说明](https://pytorch.org/docs/1.1.0/distributed.html)

> 关于测试：
> *  参考： [dsp blog]](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)
>     * 扩展能力(吞吐)，横轴卡数变多，吞吐提升
>     * 速度提升(flops) ，和MP、以及DP比吞吐速度
>     * 显存，和MP、以及DP比优势
>
> * 测试分四块：
>     * 纯DSP，Pos + g  【现有】
>     * 纯MP，去掉DSP优化 【没有，开发量一天】
>     * 纯DDP【现有】
>     * DSP+MP优化（这部分可能显存优化明显）【和第二个一致】
> * 测试数据
>     * 吞吐，直接用sample数代替，一定程度反应吞吐能力
>     * 显存，GB

## 1、 解决什么问题？  



*  faster by combination of efficiency optimizations on compute/communication/memory/IO and effectiveness optimizations on advanced hyperparameter tuning and optimizers ???
*  reduce mem consumption base on [ZeRO paper](https://arxiv.org/abs/1910.02054) /  [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
*  scalability efficient data parallelism, model parallelism, and their combination, ZeRO boost them further
*  Fast convergence for effectiveness by [LAMB](https://arxiv.org/abs/1904.00962) / [more about 1-cycle](https://www.deepspeed.ai/tutorials/1Cycle/)

## 2、 核心算法  
## 3、 用户使用方式  
## 4、 整体架构设计  
## 5、 依赖
> *  apex

### tips

> *  deepspeed 依赖 ZeRO 和 参考 NVIDIA Megatron-LM
> *  核心针对transformerbase 模型， 比如 `BERT` `GPT-2`
> *  deepspeed 同时依赖 [apex](https://github.com/NVIDIA/apex/)


### ref

> *  Turing-NLG: [ref](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
> *  ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) / [pdf](https://arxiv.org/pdf/1910.02054.pdf)
> *  [NVIDIA Megatron-LM framework](https://github.com/NVIDIA/Megatron-LM)


### 1、 What problem was solved?

目前，深度学习模型参数规模逐渐变得越来越大，尤其是NLP领域，越大的模型通常意味着更高的精度，更好的性能表现。比如曾经的Bert-large<sup> 0.3B</sup>，相对比较新的OpenAI的GPT-2<sup> 1.5B</sup>，Nvidia 的 Megatron-LM<sup> 8.3B</sup>，Google的T5<sup> 11B</sup> transformer类模型， 甚至OpenAI推出的最新的GPT 3<sup> 175B</sup> 等等，都达到了或者超越SOTA结果。但是，受限于现有GPU硬件显存限制，单个GPU无法承载如此巨大的模型训练，现有的典型分布式训练方案大都基于数据并行，数据并行分布式架构中，每个模型使用显存量相同，几乎没有优化的方案，导致现有的GPU<sup>v100 32GB显存</sup>最多支持1.4B参数模型，而对更大的参数模型训练无能为力。

针对上述问题，业界有众多的方案来完成工程化实现，比如利用模型并行方案对模型参数进行竖向切分，把参数分布到不同的机器上训练，其中典型比如`Nvidia Megatron-LM`，`Google Gpipe`等等，但是这些方案都有其自身的限制和缺陷。而由微软推出的大规模参数深度学习模型训练框架 `DeepSpeed`，是通过优化数据并行训练过程，来完成对大参数模型的训练，通过其提出的核心算法`ZeRO`<sup> Zero Redundancy Optimizer 零冗余优化</sup>, 减小训练时显存消耗，优化通讯过程，提升超大规模模型训练的扩展能力。

> *那么DeepSpeed优势是什么？ 和传统的模型并行方案相比，又有那些优势？*

### 1.1. 现有解决方案的问题
现有AI模型训练方式主要包括模型并行和数据并行方案，下面分别从这两个方案阐释相互的优缺点和取舍。

* **数据并行**<sup>  DP</sup>  
    现有的数据并行训练方式虽然**训练效率较高**，但是受限于有限的硬件显存，只能支持相对小的模型尺寸，极大的制约实际模型的训练和部署<sup>尤其是NLP类模型</sup>，同时由于训练过程中的除去模型参数的其他消耗<sup>优化器、梯度、激活占用等等</sup>，导致必须使用相对较小的`batchsize`，同样制约了训练效率，导致无法完全挖掘出卡的全部计算潜力<sup>根据OpenAI统计，大部分大规模模型训练数据并行只能使用卡的30%算力</sup>。  
    综上所述，数据并行存在明显的一些缺陷：
    1. 模型规模受限
    2. 卡的算力利用率低，计算效率差  


* **模型并行**<sup>  MP</sup>  
为了解决数据并行训练中的问题，业界提出了模型并行的方案，用来突破显存限制和提升吞吐，其中最典型的比如`Nvidia Megatron-LM`的基于模型参数和数据进行切分的模型并行方案，以及`Google Gpipe`的基于模型`Pipeline`的模型并行方案。 
    * *模型切分*<sup>  e.g. Nvidia Megatron-LM</sup>  
    `Megatron-LM`的模型切分策略可以解决大规模模型训练问题，其主要是针对模型进行竖向切分，通过精细调整模型每个层的参数切分策略和调整通讯，达成对大规模参数模型的支持<sup>主要是Transformer架构模型比如Bert、GPT等</sup>。但是，由于需要对模型结构进行精细调整，精心设计通讯策略优化通讯，导致其支持的模型种类非常有限，而且通常需要对框架代码进行深入修改，导致起扩展和兼容能力极差，同时由于不同卡间需要频分的对并行计算进行同步通讯操作，使其通信量比传统的数据并行大，导致计算效率降低。  
   综上，现有模型切分的解决方案缺陷：
        1. 通讯量大，导致计算效率降低
        2. 需要大量同步操作，增加额外开销
        3. 需要深度定制策略和代码，扩展性差  

        需要注意的是，模型切分由于可以对模型参数进行切分并部署到不同计算节点，因此在某些特定场合<sup> 单个算子参数超出显存容量</sup>，模型切分由有**不可替代**的作用<sup>比如CTR模型，包含超大(TB级别)embedding layer</sup>。

    * *模型Pipeline*<sup>  e.g. Google GPipe</sup>  
    。。。。。

### 1.2. DeepSpeed解决的问题
DeepSpeed本质上是一个基于数据并行的模型训练方案，但是通过分析数据并行模型训练中的一些特性，来对模型存储空间进行优化。  
大参数模型的数据并行训练过程中，都有一些共同特点：  
1. 显存主要被`Model states`<sup>optimizer states、grads、parameters</sup>占用
2. 剩余的显存主要被激活、临时存储和未使用的数据段占用

DeepSpeed正式围绕这两点`共识`，来对训练过程中的存储进行优化的，其通过`ZeRO`模块，提供上述两部分特性的优化方案，第一部分称之为`ZeRO-DP`，第二部分称之为`ZeRO-R`。除此之外，`DeepSpeed`还提供针对`Transformer`特有的算子优化<sup>基本类似与Nvidia 的FasterTransformer方案</sup>，用来增强`NLP`任务中的算子计算密集度，提升计算效率。

*  提出基于参数`partition`方式的`ZeRO`算法，极大降低模型训练时的显存消耗，支持更大规模参数量的模型训练<sup>10B~100B级别</sup>，解决传统分布式数据并行模型训练对模型大小的限制


### 2、Core Features and algorithm

#### 2.1. ZeRO-DP
#### 2.2. ZeRO-R
#### 2.3. Model Parallelism
    * Integration with Megatron-LM
#### 2.4. Distributed Training with Mixed Precision
    * 16-bit mixed precision
    * Single-GPU/Multi-GPU/Multi-Node

### 3、How to use?
### 4、Design arch
### 5、test
### 6、summary
### 6.1. pros & cons
### 6.2. risk


## [HugeCTR](https://github.com/NVIDIA/HugeCTR)

> ref:
>  *  [CTR prediBeginner's Guide to Click-Through Rate Prediction with Logistic Regressionction](https://turi.com/learn/gallery/notebooks/click_through_rate_prediction_intro.html)
>  *  [Mobile Ads Click-Through Rate (CTR) Prediction](https://towardsdatascience.com/mobile-ads-click-through-rate-ctr-prediction-44fdac40c6ff)


### 1、 What problem was solved?
CTR模型

CTR核心解决两个问题：
> *  CTR类模型 embedding & sparse 模型并行问题
> *  后端dnn采用数据并行



### 2、Core Features and algorithm

#### 2.1. GPU hashtable and dynamic insertion

#### 2.2. Multi-node training and enabling large embedding
> * 多节点模型中，解决大型embedding问题（模型切分，模型并行），后端dnn采用数据并行
> * 依赖NCCL & gossip 完成高速可扩展的节点内和节点间通信
> * 高性能：  三阶段pipeline，`overlap data reading from file`、`host to device data transaction`、`GPU training`

#### 2.3. Mixed precision training
> * 支持NV gpu混合精度训练
> * 混合精度训练在CMAKE中开启


### 3、How to use?
使用方式

### 4、Design arch
设计架构

### 5、test

### 6、summary

### 6.1. pros & cons

### 6.2. risk


### tips
> * deps: NCCL and gossip


### ref

> * [gossip](https://github.com/Funatiq/gossip)  提供 scatter 、gather、alltoall最优通信策略.