

## [DeepSpeed](https://github.com/microsoft/DeepSpeed)
---
### 几个核心问题：

> 评估计算力量【flops】
>  *  [pytorch 引入openai 计算量评估方法](https://discuss.pytorch.org/t/anyone-has-a-code-for-flop-calculation-of-an-epoch/49666/4)
>  *  [OpenAI 如何评估计算量](https://openai.com/blog/ai-and-compute/#fn2)


> 关于分布式：
> * DDP:  [blog 基础](https://leimao.github.io/blog/PyTorch-Distributed-Training/) 
> * DDP:  [官方说明](https://pytorch.org/docs/1.1.0/distributed.html)

1、 解决什么问题？  
*  faster by combination of efficiency optimizations on compute/communication/memory/IO and effectiveness optimizations on advanced hyperparameter tuning and optimizers ???
*  reduce mem consumption base on [ZeRO paper](https://arxiv.org/abs/1910.02054) /  [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
*  scalability efficient data parallelism, model parallelism, and their combination, ZeRO boost them further
*  Fast convergence for effectiveness by [LAMB](https://arxiv.org/abs/1904.00962) / [more about 1-cycle](https://www.deepspeed.ai/tutorials/1Cycle/)

2、 核心算法  
3、 用户使用方式  
4、 整体架构设计  
5、 依赖
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

*  faster by combination of efficiency optimizations on compute/communication/memory/IO and effectiveness optimizations on advanced hyperparameter tuning and optimizers ???
*  reduce mem consumption base on [ZeRO paper](https://arxiv.org/abs/1910.02054) /  [blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
*  scalability efficient data parallelism, model parallelism, and their combination, ZeRO boost them further
*  Fast convergence for effectiveness by [LAMB](https://arxiv.org/abs/1904.00962) / [more about 1-cycle](https://www.deepspeed.ai/tutorials/1Cycle/)


### 2、Core Features and algorithm

#### GPU hashtable and dynamic insertion


### 3、How to use?
### 4、Design arch
### 5、test
### 6、summary
### 6.1. pros & cons
### 6.2. risk


## [HugeCTR](https://github.com/NVIDIA/HugeCTR)

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