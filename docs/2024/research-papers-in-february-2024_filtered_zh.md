
> 翻译转载自: [https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) 

本次将介绍两个新的公开可用的LLM,探讨小型微调LLM的见解,以及一种新的参数高效的LLM微调技术。

上述两个LLM之所以出众,原因有几点。一个LLM (OLMo) 是完全开源的,这意味着从训练代码到数据集再到日志文件,一切都是公开共享的。

另一个LLM (Gemma) 也附带了公开可用的权重,但在多个基准测试中达到了最先进的性能,并大幅超越了类似规模的热门LLM,如Llama 2 7B和Mistral 7B。

然而，在我们探讨这个新型大语言模型架构调整的细节之前，我们首先来更详细地讨论小型大语言模型的应用前景，从下方的"Tiny Titans"论文开始。

## 1) Tiny Titans: 小型语言模型在现实世界中的会议摘要任务中能否表现出色？
在这篇[Tiny Titans论文](https://arxiv.org/abs/2402.00841)中,研究人员试图回答一个价值百万美元的问题: 拥有不到2B参数的"小型"微调LLM是否能够超越更大的、公开可用的LLM(如Llama 2 Chat)和专有LLM(如GPT-3.5)?

在讨论具体细节之前,结果总结如下表所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd55dd150-cefd-460f-abb7-e6def2255116_1600x719.png)

_本文总结了可用的大型语言模型(LLMs)和经过小型微调的大型语言模型在文本摘要性能方面的对比情况。具体信息请参考该[论文表格](https://arxiv.org/abs/2402.00841)。_

在上述表格中,"zero-shot"意味着这些模型未进一步针对会议数据和摘要进行finetune训练,而是直接使用开放可用的权重(如 Llama 2 7B Chat)或其专有API(如 GPT-3.5)。

然而,需要特别注意的是,虽然这些"zero-shot"模型未经论文作者进一步finetune,但其原始创建已涉及非常广泛的指令finetune训练,包括生成文本摘要。

现在我们可以看到,FLAN-T5-Large 在**In-Domain Dataset**类别(互联网上无法获得的真实会议数据)中表现出色,因此我们可以部分地说,经过少量finetune训练的小型 LLM 可以优于较大的可用 LLM。但是,为什么它在**QMSUM-I**数据集上的表现却远不如 GPT-3.5 和 Mixtral-8x7B?答案可能有两个原因。

首先，GPT-3.5和Mixtral可能利用了公开可用的QMSUM数据集进行了训练(由于未公开披露模型训练所使用的数据详情，我们无法确定)。

第二个可信的解释是，FLAN-T5-Large的上下文大小限制为2048个tokens,而QMSUM数据集的输入大小是它的4-5倍,如下表所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb4f4d86b-7833-4041-b00f-8010ac6fdb9f_1600x437.png)

_基于[https://arxiv.org/abs/2402.00841](https://arxiv.org/abs/2402.00841)的表格展示了不同数据集中文本长度的差异_

除了上述讨论的截断问题之外，ROUGE 得分作为自动化指标([这里有一个动手实例](https://github.com/rasbt/MachineLearning-QandAI-book/blob/main/supplementary/q19-evaluation-llms/rouge.ipynb))是否真的可靠呢?

总的来说，用于评估机器生成摘要与参考摘要之间重叠度的ROUGE评分被视为可靠，尤其是在N-gram重叠、词序列和词对匹配方面。然而，它们可能无法全面捕捉摘要在连贯性、可读性或事实准确性方面的质量，因此并非完美工具。因此，研究人员亦进行了人工评估，其结果总结于下表。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f217fcf-2336-429e-b307-5990daac64da_1322x646.png)

_基于 https://arxiv.org/abs/2402.00841 提供的信息展示了人工评估的结果。得分越高,效果越好。_

根据上表的结果,FLAN在In-Domain数据集上的表现显著优于Llama 2 7B,与GPT-3.5相当。在QMSUM-I数据集上的表现弱于前述,与我们之前观察ROUGE分数时的发现类似。

值得注意的是,用于最佳摘要参考的是GPT-4。假设GPT-3.5的预训练和指令微调方式与GPT-4相似,这可能会导致结果略有偏向,使GPT-3.5略有优势。

### **为什么小型模型表现相对较差?**

总的来说,除了在In-Domain数据集上表现良好的FLAN-T5 外,我们发现与较大的大语言模型相比,小型微调的大语言模型表现较差。

这可能有以下两个原因:一是它们的上下文窗口大小有限,导致输入数据被截断,这在生成摘要任务中可能存在问题。可以通过在训练和推理过程中扩大上下文长度来解决这个问题,这并不一定意味着模型大小的增加。二是这些模型在中间状态中存储和处理信息的能力较小。为了进一步探究这一点,我们至少需要训练具有不同上下文长度的各类大语言模型进行对比。

小型 Large Language Models (LLMs) 的选定任务是对摘要任务进行微调。不过, 对于训练大型专有 LLMs 而言, 摘要微调也是一个重要组成部分。换句话说, 我们正在将大型微调 LLMs 与小型微调 LLMs 进行比较。很有趣的是, 能够看到这种比较针对于那些并未包含在 LLMs 指令微调管线中的新颖特定领域任务。

### **Takeaways**

[Mixtral 确实很不错](https://magazine.sebastianraschka.com/p/research-papers-in-january-2024)! 而 FLAN-T5 则体积小很多, 在某些fine-tuning任务上依旧是很棒的模型。

## 2) DoRA：权重分解的低秩自适应

在[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)中，研究人员提出了一种创新性的替代方案来取代目前广泛采用的参数高效微调方法LoRA，该方法适用于大语言模型(LLMs)和视觉Transformer。我原本计划在本文中介绍这种方法,但发现这种方法如此令人兴奋,以至于我无法抑制住自己,并在几周前就实现了它。

对于那些对更多细节和论文讨论感兴趣的人,我在此撰写了一篇全面的文章和一个从头实现DoRA的指南: [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)

## 3) OLMo: 加快语言模型的科学研究

在[OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)中,"OLMo"指的是**Open Language Model**,这是一个最近发布的开源大型语言模型(有1B和7B参数版本)。值得注意的是,研究人员不仅共享了[模型权重](https://huggingface.co/allenai/OLMo-7B),还公开了所有训练细节,包括[训练代码](https://github.com/allenai/OLMo)、[训练数据](https://huggingface.co/datasets/allenai/dolma)、模型[评估代码](https://github.com/allenai/OLMo-Eval)、[日志文件](https://wandb.ai/ai2-llm/OLMo-7B/reports/OLMo-7B--Vmlldzo2NzQyMzk5)和[微调代码](https://github.com/allenai/open-instruct)。 

我强烈推荐阅读OLMo论文。虽然我希望它能包含更多见解和分析,但即便是查看一些小型架构选择和超参数配置,对我自己的实验也已经很有帮助了。以下是两个值得注意的点:

1. 他们在所有线性层禁用了bias(与Llama类似),以提高训练稳定性。
2. 相比使用标准的 LayerNorm（带有可训练的缩放和偏移参数）或 RMSNorm，他们采用了一种不包含任何可训练参数的 LayerNorm 变体。

此外，下表列出了与其他流行 LLMs 相比的主要架构差异。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fea9d4ba2-15cb-4711-bb62-12dcf45b93dd_1416x1244.png)

_OLMo架构详情(源自 OLMo 论文,_ [https://arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)_)_

此外,OLMo论文还提供了关于AdamW优化器超参数和学习率的有用详细信息,总结如下表所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fde81f535-2153-4800-a085-2063490e838e_1502x572.png)

_OLMo 优化器参数设置（来源：OLMo论文，_ [https://arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)_）_

关于学习率,他们使用了 5000 个步长（~21B tokens）的预热,然后应用了线性（而非余弦）衰减,衰减至上表所示学习率的十分之一。此外,他们将梯度裁剪到 L2-范数最大值为 1.0。

最后,值得注意的是,他们采用了线性,而非余弦学习率调度来衰减学习率,如下面的对比表所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcd75ac97-d627-4627-b521-f1c5bdd396a3_1370x688.png)
_OLMo训练参数（来源：OLMo论文，[https://arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)）_

尽管这些选择都很有趣,但它实际上与其他 Large Language Models (LLMs) 相比如何?事实证明,OLMo 与 Llama 2 和 Falcon 相当,如下表所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9ba4aaf9-52c2-44e0-8903-39edcc1a7a0f_1240x362.png)
_一个对OLMo与其他热门大型语言模型 (LLMs) 进行比较的研究,来源: [OLMo论文](https://arxiv.org/abs/2402.00838)_

虽然未对某些超参数和架构选择进行额外的消融实验，但所有日志文件均可在W&B上获得，因此我们未来几周或几个月有可能进行自己的分析。总的来说，我认为OLMo是对开源和研究社区的一个很好的贡献，我非常感谢作者分享了所有代码和训练成果。

## 4) Gemma

当然, 我们需要讨论 Google 最近推出的 Large Language Models (LLMs) Gemma。这些基于 Gemini 架构的 Gemma LLMs 有四种变体:预训练的 Gemma 2B 和 7B 基础模型,以及经过指令微调的 Gemma 2B 和 7B 模型。

除了[共享模型权重](https://huggingface.co/google/gemma-7b)之外,Google 还发布了一篇题为[Gemma: 基于 Gemini 研究和技术的开放模型](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)的技术报告,我们将在下面更详细地探讨。

### **Gemma 性能**

Gemma最显著的特点是其与其他流行和广泛使用的开源模型（如Llama 2 7B和Mistral）相比出色的性能表现，如下图所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc7a01b4a-bb23-49c1-9d6c-0ac6a152b4b2_1218x542.png)

_[Gemma技术报告](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)中的注释性能比较_

上述提到的分数是指预训练还是基于指令微调的模型变体并不完全清楚; 但我假设它们可能代表的是基于指令微调模型的性能表现。

Gemma出色的性能表现主要源于以下因素:

1. 256,000词的字典大小(相比之下, Llama的字典为32,000词);
2. 这个 6 万亿 token 的庞大训练数据集 (Llama 仅训练于其中三分之一)。

另外，在将 Gemma 集成到我们的开源 [Lit-GPT仓库](https://github.com/Lightning-AI/lit-gpt)时，我的同事和我注意到它的整体架构与 Llama 2 非常相似。让我们在下一节中进一步探讨这一方面。

### **Gemma 架构洞见**
Gemma 的一些有趣的设计选择包括：正如上文所述，其词汇表大小（以及相应的嵌入矩阵大小）非常大。下表概括了 Gemma 与我们先前讨论过的 LLaMa 2 7B 和 OLMo 7B 的架构对比。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F218c64db-2449-4a6e-b0cf-d5c5f1b75c41_1600x685.png)
_Gemma、Llama 2和OLMo的架构对比。来自[Gemma report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)的注解表_

### **模型尺寸**

值得注意的是，Gemma 2B利用了多查询注意力机制，而Gemma 7B则未采用。此外，尽管Gemma 7B的层数总共只有28层，低于Llama 2的32层，但其前馈层相对较大。然而，尽管Gemma层数较少,其参数量仍非常大。

虽然被称为Gemma 7B,但它实际上总共有93亿个参数,若考虑weight-tying, 参数量为85亿。权weight-tying味着它在输入嵌入和输出投影层共享相同的权重,类似于GPT-2和OLMO 1B(OLMO 7B在训练时未采用权重捆绑)。

### 标准化层
另一个引人注目的细节是来自论文的以下引述:

> Normalizer Location。我们对每个Transformer子层的输入和输出均进行归一化处理,这与标准做法(仅对一个进行归一化)有所不同。我们使用RMSNorm(Zhang and Sennrich, 2019)作为归一化层。

乍一看,这似乎意味着Gemma在每个Transformer块之后都有一个额外的RMSNorm层。然而,查看[official code implementation](https://github.com/keras-team/keras-nlp/blob/34a2cba28f6cb501ade97d6a9e308705d5095ec7/keras_nlp/models/gemma/rms_normalization.py#L39),发现Gemma只是采用了与GPT-2、Llama 2等其他LLM相同的标准预归一化方案,如下所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F857936c3-751e-469d-82bf-892bbbb321fe_962x1170.png)

_GPT、Llama 2 及其他大型语言模型 (LLMs) 中典型的层归一化位置: Gemma 中并未有任何新内容。(来自我的 [从零构建大型语言模型](https://www.manning.com/books/build-a-large-language-model-from-scratch) 一书中的注释图。)_

### **GeGLU激活函数**

一个与其他架构明显不同的地方是Gemma采用了GeGLU激活函数,这在2020年的论文[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)中有所提出。

GeLU,即高斯误差线性单元,是一种日益流行的激活函数,可替代传统的ReLU。GeLU之所以广受欢迎,是因为它既能引入非线性,又可允许负输入值的梯度传播,从而解决了ReLU完全阻塞负值的局限性。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfa415a6-4b27-4fab-8fed-e712fc1587a0_1298x450.png)

现在, GeGLU 是 GELU 的一种 Gated Linear Unit(带门线性单元) 变体,其激活函数被分成两部分:一部分是 Sigmoid 部分,另一部分是与第一部分输出逐元素相乘的线性投影,如下所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97b8625e-9b13-4737-b966-38db6f0391c4_1600x341.png)
正如上面所示,GeGLU类似于其他LLMs (如Llama 2和Mistral)使用的SwiGLU激活函数,不同之处在于它使用GELU作为基础激活函数而不是Swish。

这一点在查看这些激活函数的伪代码时可能更容易理解:

```
# Feedforward module with GELU (GPT-2)
x = linear(x)
x = gelu(x)
x = linear_projection(x)

# Feedforward module with SwiGLU (Llama 2)
x_1 = self.linear_1(x)
x_2 = self.linear_2(x)
x = silu(x_1) * x_2
x = linear_projection(x)

# Feedforward module with GeGLU (Gemma)
x_1 = self.linear_1(x)
x_2 = self.linear_2(x)
x = gelu(x_1) * x_2
x = linear_projection(x)
```

需要注意的是，采用SwiGLU和GeGLU的前馈模块与采用GeLU的常规前馈模块相比，确实多了一个线性层（`linear_1`和`linear_2`）。但是在GeGLU和SwiGLU前馈模块中，`linear_1`和`linear_2`通常是通过将单个线性层拆分为两部分而得到的，因此并不一定会增加参数量。

至于GeGLU是否优于SwiGLU，目前尚无消融实验结果可以确定。我猜测选择这两种不同结构可能也是为了让Gemma相较于Llama 2有些许不同。

### **其他设计选择**
此外，在为Lit-GPT添加Gemma支持时，[Andrei Aksionov发现](https://github.com/Lightning-AI/lit-gpt/pull/941)了一些其他有趣的设计选择。
例如，Gemma为RMSNorm层添加了+1的偏移量，并将嵌入层除以隐藏层维度的平方根进行归一化。后者也出现在原始的[Attention Is All You Need](https://arxiv.org/abs/1706.03762) Transformer中,但在GPT-2或OLMo等其他模型中则没有,它们也使用了权重捆绑。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4a48b78d-c5cd-4166-81a4-47dced0481ec_1322x290.png)
_来自"Attention Is All You Need"论文的摘录, [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)_

这些细节并未在论文中提及或讨论, 其重要性也不太清楚。

可能的解释是, 尽管线性层和嵌入层执行相同的功能(见我关于[嵌入层和线性层](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb)的撰写), 但初始权重的量级不同。由于 Google 的研究人员更偏好使用 TensorFlow 进行实验, 乘以嵌入维度的平方根也许是一种将权重调整到更合理的量级的方法, 如我在下面的示例代码中所示。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbc39e6fc-6b60-4006-801b-857cab1af0de_1600x1188.png)

将嵌入维度的平方根乘以权重,可以得到与标准线性层中权重相同量级的权重。

总的来说，Gemma是一个不错的补充,扩充了公开可用的大型语言模型(LLM)集合。这个7B模型似乎是一个非常强大的模型,有可能在实际应用中取代Llama 2和Mistral。

此外，由于我们已拥有约70亿模型的广阔开放集合,Gemma 2B模型更有意义的是它可轻松在单GPU上运行。值得关注的是,它与同样规模为2.7B的Phi-2模型相比会有何表现。

如果你想通过上述提到的Lit-GPT实现在实践中使用Gemma，我在[这里](https://lightning.ai/lightning-ai/studios/understanding-using-and-finetuning-gemma)创建了一个Studio环境。
![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00fe2106-41da-4449-9f56-7ce300bf90dc_1170x1180.png)
通过 Lit-GPT Studio ([https://lightning.ai/lightning-ai/studios/understanding-using-and-finetuning-gemma](https://lightning.ai/lightning-ai/studios/understanding-using-and-finetuning-gemma)) 使用 Gemma

## 2月份其他有趣的研究论文

**Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models** by De, Smith, Fernando _et al._ (29 Feb), [https://arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427)

* 本文介绍了Hawk和Griffin，这是一个循环神经网络LLM和一个结合循环神经网络元素与局部注意力的LLM混合架构，提出了一种新的高效替代transformer-based LLM的方案。

**When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method** by Zhang, Liu, Cherry, and Firat (27 Feb), [https://arxiv.org/abs/2402.17193](https://arxiv.org/abs/2402.17193)

* 本研究系统探讨了 model size、预训练数据规模、微调参数规模和微调数据规模等缩放因素如何影响大型语言模型(LLM)的微调性能。针对LLM规模超过微调数据规模的场景, 研究比较了全模型微调与参数高效微调(包括提示微调和LoRA)的性能。

**Sora Generates Videos with Stunning Geometrical Consistency** by Li, Zhou, Zhang _et al._ (27 Feb), [https://arxiv.org/abs/2402.17403](https://arxiv.org/abs/2402.17403)

* 本文介绍了一个评估 Sora 模型生成视频与现实世界物理保真度的基准测试。该方法通过将生成的视频转换为 3D 模型,并以 3D 重建的准确性作为指标来评估其对物理原理的遵守情况。研究发现,与其他文本到视频模型(如 Pika)相比,Sora 的表现非常出色。

**\* The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** by Ma, Wang, Ma _et al._ (27 Feb), [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)

* 本研究介绍了一种 1 位 LLM 变体(仅支持值 -1、0 和 1), 该变体在推理期间的困惑度和下游任务性能与采用传统 16 位精度的 LLM 相当。

**Genie: Generative Interactive Environments** by Bruce, Dennis, Edwards _et al._ (23 Feb), [https://arxiv.org/abs/2402.15391](https://arxiv.org/abs/2402.15391)

* Genie 是一个开创性的 11B 参数生成式交互环境(基于时空 Transformer)，从互联网视频中无监督训练而成，能够从文本、图像和素描中创造出无穷变量、可控动作的虚拟世界。

**\* Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs** by Ahmadian, Cremer, Galle, _et al._ (22 Feb), [https://arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740)

* 这项研究表明，相较于近端策略优化(Proximal Policy Optimization, PPO)，更简单的REINFORCE风格优化方法在通过来自人类反馈的强化学习(Reinforcement Learning from Human Feedback, RLHF)进行AI对齐的大型语言模型中更加高效和有效。

**TinyLLaVA: A Framework of Small-scale Large Multimodal Models** by Zhou, Hu, Weng, _et al._ (22 Feb), [https://arxiv.org/abs/2402.14289](https://arxiv.org/abs/2402.14289)

* TinyLLaVA框架表明,使用高质量数据和优化训练的小型 Large Multimodal Models(LMMs)可以与更大的模型相媲美或超越,其最佳模型 TinyLLaVA-3.1B 已经超过了现有的7B模型。

**Large Language Models for Data Annotation: A Survey** by Tan, Beigi, Wang _et al._ (21 Feb), [https://arxiv.org/abs/2402.13446](https://arxiv.org/abs/2402.13446)

* 本文探讨了像GPT-4这样的大型语言模型(LLMs)在自动化数据标注这一劳动密集型过程方面的潜力,重点关注LLM生成的标注在数据标注、评估以及从这些标注中学习方面的应用。

**\* LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** by Ding, Zhang, Zhang, _et al._ (21 Feb), [https://arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)

* LongRoPE是一种新的方法,该方法通过对位置插值的改进和渐进式扩展策略,将预训练的大型语言模型的上下文窗口扩展到2,048,000个token,同时保持跨上下文大小的性能,只需最少的微调。

**YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information** by Wang, Yeh, and Liao (21 Feb), [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)

* 这项研究提出了解决信息瓶颈和深度监督机制不适宜性的方案,并在 MS COCO 数据集上提出了相比 YOLOv8 具有更高效和更佳性能的 YOLOv9。

**Neural Network Diffusion** by Wang, Xu, Zhou, _et al._ (20 Feb), [https://arxiv.org/abs/2402.13144](https://arxiv.org/abs/2402.13144)

* 本研究展示了扩散模型(traditionally 用于图像和视频生成)如何应用于生成高性能神经网络参数。扩散模型是一种新兴的生成模型,可用于各种创造性任务,如图像、音乐和文本生成。本研究将此方法应用于神经网络参数生成,证明了其在这一领域的潜力。

**\* LoRA+: Efficient Low Rank Adaptation of Large Models** by Hayou, Ghosh, and Yu (Feb 19),  [https://arxiv.org/abs/2402.12354](https://arxiv.org/abs/2402.12354)

* 本文介绍了LoRA+, 这是对原始Low Rank Adaptation (LoRA)方法的改进。通过为adapter矩阵A和B采用不同的学习率, 可增强特征学习, 在不增加计算成本的情况下, 可获得1-2%的性能提升和最多2倍的微调速度。

**Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs** by Boizard, Haddad, Hudelot, and Colombo (19 Feb), [https://arxiv.org/abs/2402.12030](https://arxiv.org/abs/2402.12030)

* 该论文介绍了 Universal Logit Distillation 损失函数,该函数可以实现不同结构和 Tokenizer 的大型语言模型之间进行有效的知识蒸馏,从而克服了共享 Tokenizer 的限制。

**AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling by** Zhan, Dai, Ye, and Zhou (19 Feb), [https://arxiv.org/abs/2402.12226](https://arxiv.org/abs/2402.12226)

* AnyGPT 是一种多模态语言模型,它通过离散表示无缝集成了语音、文本、图像和音乐,实现了多样化的任意输入输出的多模态交互,而不会改变大型语言模型的核心架构。

**\* Reformatted Alignment** by Fan, Li, Zou, and Li (19 Feb), [https://arxiv.org/abs/2402.12219](https://arxiv.org/abs/2402.12219)

* 本文介绍了 ReAlign，这是一种通过简单的重格式化方法来提升大型语言模型（LLMs）微调数据质量，从而使其更好地与人类价值观保持一致的方法。

**LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration** by Zhao, Zu, Xu, et al (18 Feb), [https://arxiv.org/abs/2402.11550](https://arxiv.org/abs/2402.11550)

* LongAgent 通过多智能体协作和成员之间的通信机制 [**提升**] 了 LLMs 对长文本的处理能力，在文本检索和多跳问题回答等任务上 [**优于**] GPT-4 等模型。

**Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning** by Xu, Feng, Shao, et al. (2024), [https://arxiv.org/abs/2402.11690](https://arxiv.org/abs/2402.11690)

* 该研究介绍了 Vision-Flan，这是一个多样化的视觉指令微调数据集,以及用于视觉语言模型的两阶段指令微调框架。该框架通过结合 Expert 和 GPT-4 合成的数据来解决泛化性差和偏差等问题。


**OneBit: Towards Extremely Low-bit Large Language Models** by Xu, Han, Yang, et al. (17 Feb), [https://arxiv.org/abs/2402.11295](https://arxiv.org/abs/2402.11295)

* 该论文介绍了 OneBit，这是一个针对 LLMs 的 1 位量化感知训练框架。通过新的参数表示和初始化方法，实现了显著的存储和计算效率，同时性能损失很小。

**FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models** by Bhatia, Nagoudi, Cavusoglu, and Abdul-Mageed (16 Feb), [https://arxiv.org/abs/2402.10986](https://arxiv.org/abs/2402.10986)

* FinTral 是一款针对金融分析进行优化的多模态 LLM 套件,基于 Mistral-7b 构建,并通过特定领域的训练和基准测试得到增强,在关键任务中优于 ChatGPT-3.5 和 GPT-4 - 这是一个很好的 AI 金融应用案例研究。

**Generative Representational Instruction Tuning by Muennighoff**, Su, Wang, et al. (15 Feb), [https://arxiv.org/abs/2402.09906](https://arxiv.org/abs/2402.09906)

* GRIT是一种全新的训练方法，它使得LLM(大型语言模型)GritLM在遵循指令的情况下，能够在生成任务和Embedding任务上都取得出色的表现。

**Recovering the Pre-Fine-Tuning Weights of Generative Models** by Horwitz, Kahana, and Hoshen (15 Feb), [https://arxiv.org/abs/2402.10208](https://arxiv.org/abs/2402.10208)

* 该论文介绍了 Spectral DeTuning 技术，这种方法能够从大规模模型（如 Stable Diffusion 和 Mistral）的 fine-tuned 模型中恢复 pre-fine-tuning 的权重。

**BASE TTS: Lessons From Building a Billion-Parameter Text-to-Speech Model on 100K Hours of Data** by Lajszczak, Cambara, Li, _at al._ (15 Feb), [https://arxiv.org/abs/2402.08093](https://arxiv.org/abs/2402.08093)

* 由 Amazon 研究人员开发的 BASE TTS 是一种新型文本转语音模型，它通过使用包含十亿个参数的 Transformer 架构对 100K 小时的数据进行训练，实现了前所未有的语音自然性。

**Transformers Can Achieve Length Generalization But Not Robustly** by Zhou, Alon, Chen, _et al._ (14 Feb) [https://arxiv.org/abs/2402.09371](https://arxiv.org/abs/2402.09371)

* 这篇论文探讨了在语言模型中进行长度推广的挑战,证明了标准的Transformer模型能够利用特定的数据格式和位置编码,将序列长度推广到其训练输入的2.5倍。不过,这种能力对诸如权重初始化和数据顺序等因素高度敏感。

**\* DoRA: Weight-Decomposed Low-Rank Adaptatio**n by Liu, Wang, Yin _et al._ (14 Feb), [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)

* DoRA是标准低秩适应(LoRA)的一种改进版本,通过将预训练权重分解为幅度和方向,可以弥补参数高效微调方法(如LoRA)与完整微调之间的准确性差距,实现更有效的更新。

**Mixtures of Experts Unlock Parameter Scaling for Deep RL** by Obando-Ceron, Sokar, Willi, _et al._ (13 Feb), [https://arxiv.org/abs/2402.08609](https://arxiv.org/abs/2402.08609)

* 这篇论文表明，将 Mixture-of-Expert (MoE) 模块（尤其是 Soft MoE）集成到基于价值的强化学习网络中，可使模型在规模扩大时更有效地扩展，为在强化学习领域建立扩展定律指明了道路。

**World Model on Million-Length Video And Language With RingAttention** by Liu, Yan, Zaharia, and Abbeel (13 Feb), [https://arxiv.org/abs/2402.08268](https://arxiv.org/abs/2402.08268)

* 该工作提出通过使用最近的技术 如 RingAttention, 对大规模的长视频和语言序列数据集进行神经网络训练。

**Suppressing Pink Elephants with Direct Principle Feedback** (12 Feb), [https://arxiv.org/abs/2402.07896](https://arxiv.org/abs/2402.07896)

* 这项研究介绍了Direct Principle Feedback (DPF)这一新方法,用于实时调整大型语言模型的响应,并展示了其在成功指导模型避免特定话题的"粉色大象问题"上的能力。

**Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping** Wang, Ma, Meng, _et al._ (12 Feb), [https://arxiv.org/abs/2402.07610](https://arxiv.org/abs/2402.07610)

* 该研究通过 Step-On-Feet Tuning (SOFT) 探索了在 LLMs 中的 multi-time bootstrapping self-alignment 技术,该方法通过利用迭代对齐和优化的训练序列,相比单步方法可以提高模型性能。

**Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model** by Ustun, Aryabumi, Yong, _et al._ (12 Feb), [https://arxiv.org/abs/2402.07610](https://arxiv.org/abs/2402.07610)

* 本研究介绍了 Aya，一款在 101 种语言中均达到精通水平的多语言生成式语言模型。

**Scaling Laws for Fine-Grained Mixture of Experts** by Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, _et al._ (12 Feb), [https://arxiv.org/abs/2402.07871](https://arxiv.org/abs/2402.07871)

* 这项研究探索了 Mixture of Experts (MoE) 模型的缩放特性,引入了"粒度"这一新的超参数来调整专家规模,并展示了 MoE 模型优于密集型 Transformer 的优势。

**Policy Improvement using Language Feedback Models** by Zhong, Misra, Yuan, and Cote (12 Feb), [https://arxiv.org/abs/2402.07876](https://arxiv.org/abs/2402.07876)

* 通过使用LLMs对口述视觉轨迹提供反馈,Language Feedback Models (LFMs)可以提高模仿学习效果。相比传统方法,LFMs在任务完成和适应新环境方面表现优异,并能提供人类可解释的反馈,以验证期望的行为。

**ODIN: Disentangled Reward Mitigates Hacking in RLHF** by Chen, Zhu, Soselia _et al._ (11 Feb), [https://arxiv.org/abs/2402.07319](https://arxiv.org/abs/2402.07319)

* 该研究通过开发细致入微的评估协议和改进的REWARD模型,共同训练两个头部,以优先考虑内容而非长度,大幅减少长度偏差并提升策略效果。

**The Boundary of Neural Network Trainability is Fractal** by Dickstein (9 Feb), [https://arxiv.org/abs/2402.06184](https://arxiv.org/abs/2402.06184)

* 本研究发现神经网络训练中存在类分形[**与**]边界,突出了训练动力学对微小超参数调整的极端敏感性,涵盖了广泛的配置和规模。

**Buffer Overflow in Mixture of Experts** by Hayes, Shumailov, and Yona (8 Feb), [https://arxiv.org/abs/2402.05526](https://arxiv.org/abs/2402.05526)

* 该研究显示, Mixture of Experts (MoE) 模型容易受到基于专家路由策略的依赖跨批次的攻击影响, 其中恶意查询可以影响同一批次内正常查询的输出。

**Direct Language Model Alignment from Online AI Feedback** by Guo, Zhang, Liu, et al. (7 Feb), [https://arxiv.org/abs/2402.04792](https://arxiv.org/abs/2402.04792)

* 本论文提出了一种在线反馈模型训练方法,该方法通过使用LLM的实时评估,超越了像DPO和RLHF等基于偏好的直接对齐(DAP)方法。

**Grandmaster-Level Chess Without Search** by Ruoss, Deletang, and Medapati (7 Feb), [https://arxiv.org/abs/2402.04494](https://arxiv.org/abs/2402.04494)

* 本文由Google DeepMind提出,介绍了一个"小型"的270M参数Transformer模型,该模型在10百万盘国际象棋数据上进行了训练,在国际象棋性能方面超越了AlphaZero和GPT-3.5-turbo-instruct网络。

**Self-Discover: Large Language Models Self-Compose Reasoning Structures** by Zhou, Pujara, Ren, _et al._ (6 Feb), [https://arxiv.org/abs/2402.03620](https://arxiv.org/abs/2402.03620)

* SELF-DISCOVER赋予了LLMs自主创建推理策略的能力，显著提升了它们的问题解决效率和跨模型适用性，可能更贴近人类的推理方式。

**Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models** by Guo, Chen, Wang _et al._ (6 Feb), [https://arxiv.org/abs/2402.03749](https://arxiv.org/abs/2402.03749)

* 本论文通过一种自适应置信损失的知识蒸馏方法,探讨了视觉基础模型中从弱到强的泛化能力。研究发现,弱模型可以有效地增强强模型的性能,这标志着AI在视觉任务方面取得了重大进步。

**MOMENT: A Family of Open Time-series Foundation Models** by Goswami, Szafer, Choudhry, _et al._ (6 Feb), [https://arxiv.org/abs/2402.03885](https://arxiv.org/abs/2402.03885)

* MOMENT 引入了一种新的面向通用目的时间序列分析的方法,利用开源基础模型来解决缺乏统一的时间序列数据集以及多数据集训练复杂性等挑战。通过创建 Time-series Pile 并设计低监督环境下的基准测试,对模型性能进行评估。

**Scaling Laws for Downstream Task Performance of Large Language Models by Isik, Ponomareva, Hazimeh**, _et al._ (6 Feb), [https://arxiv.org/abs/2402.04177](https://arxiv.org/abs/2402.04177)

* 该研究探讨了预训练数据的规模和相关性如何影响LLMs的机器翻译效果。研究发现, 数据与目标任务[**与**]训练语料库对齐可以改善翻译结果, 而数据[**与**]任务不匹配可能导致结果不一致。

**A Phase Transition Between Positional and Semantic Learning in a Solvable Model of Dot-Product Attention** by Cui, Behrens, Krzakala, and Zdeborova (6 Feb), [https://arxiv.org/abs/2402.03902](https://arxiv.org/abs/2402.03902)

* 本研究探讨了点积 Attention 层如何学习专注于数据中的位置或含义,揭示了在数据充足的情况下,这些层可以通过从位置性 Attention 向语义性 Attention 机制的过渡,从而超越线性模型。

**MobileVLM V2: Faster and Stronger Baseline for Vision Language Model** by Chu, Qiao, Zhang _et al._ (6 Feb), [https://arxiv.org/abs/2402.03766](https://arxiv.org/abs/2402.03766)

* MobileVLM V2引入了小型高效的视觉语言模型，1.7B版本的性能可媲美甚至超越3B规模的模型，而3B版本也超越了7B+规模的模型。

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** by Shao, Wang, Zhu _et al._ (5 Feb), [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

* DeepSeekMath 7B 是一个在 120B 个与数学相关的 tokens 上预训练的大型语言模型(LLM)。通过利用网页数据以及引入 Group Relative Policy Optimization(一种替代 PPO 的方法)用于数学推理,在 MATH 基准测试中取得了 51.7% 的分数,接近领先模型 Gemini-Ultra 和 GPT-4 的性能。

**More Agents Is All You Need** by Li, Zhang, Yu, Fu, and Ye (3 Feb), [https://arxiv.org/abs/2402.05120](https://arxiv.org/abs/2402.05120)

* 该研究表明,通过在多个LLM(Large Language Model,大型语言模型)中采用简单的多数投票集成方法,可以提升LLM的性能表现。这种方法不仅补充了现有的方法,在较为困难的任务中尤为突出。

**FindingEmo: An Image Dataset for Emotion Recognition in the Wild** by Mertens, Yargholi, Op de Beeck _at el._ (2 Feb), [https://arxiv.org/abs/2402.01355](https://arxiv.org/abs/2402.01355)

*   FindingEmo是一个新的 Emotion Recognition 数据集,包含 25k 张图像数据,针对自然场景中复杂的多人场景进行了情绪识别标注,数据和源代码已公开共享。

**\* LiPO: Listwise Preference Optimization through Learning-to-Rank** by Liu, Qin, Wu, _et al._ (2 Feb), [https://arxiv.org/abs/2402.01878](https://arxiv.org/abs/2402.01878)

* 这项工作引入了 Listwise Preference Optimization (LiPO)，以将对齐视为一个 listwise 排序问题的方式来使 LLMs 与人类反馈保持一致。结果显示 LiPO 优于当前的如 DPO 等策略优化方法。

**\* Repeat After Me: Transformers are Better than State Space Models at Copying** by Jelassi, Brandfonbrener, Kakade, and Malach (1 Feb), [https://arxiv.org/abs/2402.01032](https://arxiv.org/abs/2402.01032)

* 本文展示了虽然状态空间模型在推理时效率方面有优势,但由于其固定大小的潜在状态存在固有局限性,在需要输入上下文复制的任务中,它们无法与Transformer模型相匹。

**\* Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?** by Fu, Laskar, Khasanonva _et al._ (Feb 1), [https://arxiv.org/abs/2402.00841](https://arxiv.org/abs/2402.00841)

* 这项研究发现 Compact Large Language Models 如 FLAN-T5 在执行特定任务(如会议总结)方面能够与更大型模型相匹配或者超越,在效率和性能方面具有优势。这种成本效益较高的替代方案在实际部署中具有实用价值。

**\* OLMo: Accelerating the Science of Language Models** by Groeneveld, Beltagy, Walsh, _et al._ (1 Feb),  [https://arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)

* 本技术报告介绍了 OLMo，这是一款完全开放的 LLM，以及其完整框架，包括训练数据、训练和评估代码。

**Efficient Exploration for LLMs** by by Dwaracherla, Asghari, Hao, and Van Roy (1 Feb), [https://arxiv.org/abs/2402.00396](https://arxiv.org/abs/2402.00396)

* 本研究论文展示，通过有效探索生成针对人类反馈的查询方式，可基于收到的反馈持续优化奖励模型，从而以较少的查询大幅提升 LLMs 的性能。


