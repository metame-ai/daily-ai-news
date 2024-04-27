Apple release OpenELM
============================================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-apples-openelm-beats-olmo-with-50-of-its/](https://buttondown.email/ainews/archive/ainews-apples-openelm-beats-olmo-with-50-of-its/) 

[Apple的AI崛起](https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/)在WWDC之前持续高涨。我们先前曾介绍过[OLMo](https://buttondown.email/ainews/archive/ainews-ai2-releases-olmo-the-4th-open-everything/)，现在看来OpenELM是Apple首次发布的[真正开放的大型语言模型](https://arxiv.org/abs/2404.14619)([权重](https://huggingface.co/apple/OpenELM)、[代码](https://github.com/apple/corenet))，其中分享了一些在高效架构方向的创新研究。

![image.png](https://assets.buttondown.email/images/3bd4b772-df2f-46b7-8318-2cc230b7eb46.png?w=960&fit=max)

如[Sebastian Raschka](https://twitter.com/rasbt/status/1783480053847736713/photo/1)所言:

> 我们就从最有趣的细节开始吧:
>
> OpenELM 提供 4 种相对较小、便利的尺寸: 270M、450M、1.1B 和 3B。
> 尽管 OpenELM 使用了 2 倍更少的 tokens 进行训练，但其性能仍略优于 OLMo。
> 主要的架构调整是采用了分层缩放策略

但是:

> 他们从各种公开可用数据集(RefinedWeb、RedPajama、The PILE和Dolma)中抽取了相对较小的1.8T token子集。这个子集比用于训练OLMo的Dolma小2倍。这种子采样的基本原理和采用的具体标准是什么?

![image.png](https://assets.buttondown.email/images/5a0bcc71-6f46-41a3-a34b-6efff203c64d.png?w=960&fit=max)

这种分层缩放来自于[DeLight](https://arxiv.org/abs/2008.00623),这是一篇2021年的论文,它将标准注意力机制的层数加深2.5-5倍,但通过参数数量匹配了2-3倍更大的模型。这似乎存在矛盾,但作者描述了主要技巧是在输入与输出之间调整深度,而非保持统一。

![image.png](https://assets.buttondown.email/images/64a3ecf6-fbca-4816-9233-f4100454aca8.png?w=960&fit=max)

![image.png](https://assets.buttondown.email/images/a70b5ba1-00bb-482d-a4a4-f1027eec0266.png?w=960&fit=max)

* * *


AI Reddit Recap
===============


**LLaMA Developments**

*   **LLaMA 3 将上下文增加到 160K+ 个 tokens**：在 /r/LocalLLaMA 中，LLaMA 3 将上下文长度增加到[**超过 160K 个 tokens，同时保持完美的回忆**](https://www.reddit.com/r/LocalLLaMA/comments/1ccqmjz/llama_3_now_with_160k_context/)。评论者指出这一点非常出色，但需要强大的消费级硬件才能在本地以良好的速度运行。Meta 的 Llama 3 已被下载超过 120 万次，在 Hugging Face 上有超过 600 个衍生模型。
* 首个 LLama-3 8B-Instruct 模型,具有 262K 上下文长度已发布:在 /r/LocalLLaMA 上,首个拥有[**超过 262K 上下文长度的 LLama-3 8B-Instruct 模型已在 Hugging Face 上发布**](https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/),可进行高级推理。
* 在 /r/LocalLLaMA 上的比较显示, [**量化的 Llama 3 70B IQ2_XS 优于未压缩的 Llama 3 8B f16 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1cda0fv/llama_3_8b_f16_vs_llama_3_70b_q2/)。对于 32GB VRAM 用户而言, 70B IQ3_XS 版本被认为是最佳选择。
*   **新论文比较AI对齐方法**：在 /r/LocalLLaMA 上，一篇新论文对比了 DPO 与其他对齐方法，发现 [**KTO 在大多数基准测试中表现最佳，且对齐方法对训练数据量很敏感**](https://www.reddit.com/r/LocalLLaMA/comments/1ccz84a/insights_into_alignment_dpo_and_its_variants/)。

**AI Ethics & Regulation**

* 前谷歌CEO Eric Schmidt在/r/singularity上警告称,[**开源AI模型为不法分子和中国提供了有风险的能力**](https://www.reddit.com/r/singularity/comments/1ccyqkr/former_google_ceo_eric_schmidt_warns_that_open/)。许多人认为这是大型科技公司试图遏制竞争的行为,并指出中国可能有能力开发强大的AI模型而无需依赖开源。
* 美国提案旨在终止云计算的匿名使用: 在/r/singularity上, [**美国提案寻求实施"了解您的客户"要求,以结束云计算的匿名使用**](https://www.reddit.com/r/singularity/comments/1ccr2ub/us_know_your_customer_proposal_will_put_an_end_to/)。

*   **巴尔的摩教练涉嫌使用 AI 进行诽谤**：据悉,在 /r/OpenAI 上,一名巴尔的摩教练[**使用 AI 语音克隆技术试图让一名高中校长被解雇,通过生成假的种族主义音频**](https://www.reddit.com/r/OpenAI/comments/1cd5h9c/baltimore_high_school_athletic_director_used_ai/)。

**Hardware Developments**

* TSMC宣布推出1.6纳米制程节点:在/r/singularity上,TSMC宣布了[**1.6纳米制程节点并提供背面供电**](https://www.reddit.com/r/singularity/comments/1ccr4hy/tsmc_unveils_16nm_process_technology_with/),使未来数年内硬件能够持续实现指数级进步。
* 在/r/singularity上,德国研究人员开发出了**超薄、柔性的太阳能电池**,使小型无人机能够在运行过程中自行[充电](**https://www.reddit.com/r/singularity/comments/1ccr6aq/german_researchers_have_developed_a_solar_cell/**)。
* 美光获得 CHIPS 法案 60.1 亿美元资金,用于在纽约和爱达荷州建立半导体制造设施。

* * *

AI Twitter Recap
================


**Meta Llama 3 Release and Impact**

*   **训练优化**：Meta正在快速推进优化工作,Llama 3 70B的训练速度提高了18%,Llama 3 8B的训练速度提高了20%。 ([@svpino](https://twitter.com/svpino/status/1783888989025431933))
* 该社区通过结合PoSE、持续预训练和RoPE缩放,将Llama 3 8B的上下文从8k扩展至近100k tokens。([@winglian](https://twitter.com/winglian/status/1783842736833016289))
*  **推理加速**: Colossal-Inference 现已支持 Llama 3 推理加速功能, 为 8B 和 70B 模型提高约 20% 的效率。([@omarsar0](https://twitter.com/omarsar0/status/1783895931043111088))
*   **Llama 3 70B在LMSYS排行榜上的英语查询任务中与其他模型并列第一。**([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783570318230978783))

**Phi-3 Model Release and Reception**

*   **过度拟合基准测试**：有观点认为 Phi-3 在公开基准评测中过度拟合，但与 Llama-3 8B 等模型相比在实际应用中表现不如人意。([@svpino](https://twitter.com/svpino/status/1783556635543339310)，[@abacaj](https://twitter.com/abacaj/status/1783898711623352686))
* 作为一个从根本上不异于常规模型的Phi-3，可能会出现令人惊讶的正面和负面结果。([@srush_nlp](https://twitter.com/SebastienBubeck/status/1783885843943616524))

**Extending LLM Context Windows**

*   **PoSE 技术**：Positional Skip-wisE (PoSE) 方法在训练过程中模拟长输入,以增加上下文长度,从而支持 Llama 3 扩展至 128k tokens。([@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1783574428858696161))
*   **Axolotl和Gradient AI**：类似Axolotl的工具和Gradient AI的方法正在为Llama等模型提供上下文扩展至160k+个token的功能。([@winglian](https://twitter.com/winglian/status/1783469196011016696), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783736130321408011))

**Cohere Toolkit Release**

*   **企业重点**: Cohere 发布了一个工具包,用于加速企业中的 LLM 部署,针对使用私有数据和本地代码解释器的安全 RAG。([@aidangomez](https://twitter.com/aidangomez/status/1783533461401227563))
* 该工具包的组件可部署至任何云端环境并重复利用以构建应用程序。([@aidangomez](https://twitter.com/aidangomez/status/1783533465960378561), [@aidangomez](https://twitter.com/aidangomez/status/1783533471777935433))

**OpenAI Employee Suspension and GPT-5 Speculation**

*   **Sentience Claims**: OpenAI 员工声称 GPT-5 具有感知能力,该员工随后遭到 Twitter 的账号暂停处理。([@bindureddy](https://twitter.com/bindureddy/status/1783847600824995850))
*   **炒作生成**：OpenAI 被视为围绕 AGI 和 AI 意识主张的炒作引擎,即使竞争对手以较低成本与 GPT-4 相匹敌。([@bindureddy](https://twitter.com/bindureddy/status/1783852748636905716))
*   **Agent 能力**：有人认为 GPT-5 将是一种"Agent GPT"，这是基于在语言模型之上增加 Agent 基础设施而带来的性能提升。([@OfirPress](https://twitter.com/OfirPress/status/1783870394581074110))

**Other Noteworthy Topics**

* 关于AI峰会董事会缺乏 Diverse 代表性以应对权力集中风险的担忧。([@ClementDelangue](https://twitter.com/ClementDelangue/status/1783882237764633052))
* OpenAI与Moderna的合作,可视作传统企业采用生成式AI的积极信号。([@gdb](https://twitter.com/gdb/status/1783529202974687527), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783533728846827681))
* Apple开源的设备端语言模型虽然展示了较差的性能表现, 但却提供了有价值的架构和训练细节。([@bindureddy](https://twitter.com/bindureddy/status/1783635037365436462), [@rasbt](https://twitter.com/rasbt/status/1783480053847736713))

* * *

AI Discord Recap
================


**Extending LLM Context Lengths**

*   **Llama 3性能和上下文长度创新**：围绕**Llama 3的功能**展开了讨论,对其与**GPT-4**相比在代码记忆和配置方面存在一些不同看法。然而,采用**PoSE（Positional Skip-wisE）**等技术将Llama 3的**上下文长度扩展至8B模型的96k个token**,以及持续使用300M个生成的token进行预训练,引发了广泛关注,详见此[推特](https://x.com/winglian/status/1783456379199484367)。
*   [EasyContext项目](https://github.com/jzhang38/EasyContext)旨在以最低的硬件要求, 将LLM上下文长度外推至**1百万个tokens**。
优化LLM培训和部署


**Optimizing LLM Training and Deployment**

* Nvidia的[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)用于**内核分析** (kernel profiling)来优化LLM训练的CUDA代码。

* 人们对**微调大型语言模型**以获得特定领域的改进益处深感兴趣,例如**[Meditron](https://arxiv.org/abs/2311.16079)**用于医疗应用。讨论还涉及了使用**[Argilla's Distilabel](https://github.com/argilla-io/distilabel)**等工具进行数据合成策略,以及多文档、长上下文微调的挑战。成本和性能之间的权衡也受到争论,例如投入[$2,368用于4个epoch与$41,440用于50个epoch](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028)可能只会带来微小的改善。
* PyTorch 推出了 [Torchtitan](https://github.com/pytorch/torchtitan)，这是一个专门用于从头开始训练 Large Language Model (LLM) 的库。
*   **CUDA 优化深入探讨**：CUDA 开发人员使用[NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)等工具深入研究内核分析，探讨了 **memory coalescing** 和约 128 字节的 **burst sizes** 相关问题，并就 **low-bit quantization** 方法的效率进行了讨论。对话还涉及 PyTorch 2.3.0 中 **flash attention 兼容性** 问题，以及 PyTorch AO 支持 **自定义 CUDA 扩展** 对性能调优的影响。

**Open-Source LLM Ecosystem Expansion**

* 苹果推出[OpenELM](https://huggingface.co/apple/OpenELM)系列高效开源语言模型,从270M到3B参数不等,令AI界颇感意外。这标志着苹果从传统的专有模式转向开源,其中270M模型在Hugging Face上迅速引起关注。
* [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B)引入了利用Mistral对预训练的医疗领域大型语言模型进行微调的方法。
* Mozilla的[llamafile project](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/)可以以高性能本地分发和运行LLMs(大型语言模型)。
* Dify 作为一个[open-source LLM app development platform](https://github.com/langgenius/dify)应运而生,它融合了 AI 工作流和模型管理功能。

**Evaluating and Benchmarking LLMs**

* 在[Judgemark基准测试](https://eqbench.com/judgemark.html)中，**Llama-3-70b**显示了可用于微调**disco-judge**应用程序的前景。
* 围绕**验证损失**作为评判大型语言模型性能指标的有效性的讨论。
* [低成本语言模型调研](https://arxiv.org/abs/2404.11160)评估了在Python代码生成任务上的CPU友好型大语言模型。
* 围绕 **Nightshade's** 自编码器功能的透明度以及公开发布研究结果的必要性进行了讨论。

* * *

