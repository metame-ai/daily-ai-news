Perplexity: 最新的AI独角兽
============================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-perplexity/](https://buttondown.email/ainews/archive/ainews-perplexity/) 

仅在[Series B](https://buttondown.email/ainews/archive/ainews-142024-jeff-bezos-backs-perplexitys-520m/)后3个月,Perplexity再次翻番估值,进行了[Series B-1]融资,绝大部分投资者名单与上次相同,但 Daniel Gross 未与 Nat Friedman 共同牵头。Dan似乎与该公司有特殊关系 - Aravind 分享了[2022年12月Dan对产品反馈的电子邮件](https://x.com/AravSrinivas/status/1782785662607114365)。

![image.png](https://assets.buttondown.email/images/60694bbc-7fdd-4bb0-8a9a-928b03a06a30.png?w=960&fit=max)

* * *


AI Reddit Recap
===============


**Llama 3 Variants and Optimizations**

* 在 /r/LocalLLaMA 中, Llama-3-8B 的上下文长度[**已扩展至16K个token**](https://huggingface.co/mattshumer/Llama-3-8B-16K), 使其原始的上下文窗口大幅增加。
* 基于Llama 3的[**LLaVA模型**](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1)由XTuner团队在Hugging Face上发布,在各种基准测试中明显优于Llama 2。
*   **BOS Token 提醒**: 在 /r/LocalLLaMA 中, 一篇 [**公告提醒用户在微调 Llama 3 模型时确保训练设置添加了 BOS token**](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/) 以避免出现 inf grad_norm 或更高损失的问题。
*   **Special Token Embedding Adjustments**: 针对 Llama-3-8B 中[**未训练的特殊 token embedding**](https://huggingface.co/astronomer/Llama-3-8B-Special-Tokens-Adjusted)进行了调整, 并已在 Hugging Face 上共享, 以解决由于零值引起的微调问题。
* 在 /r/LocalLLaMA 中，发布了[适用于网页浏览和用户交互的 Llama-3-8B-Web action model](https://www.reddit.com/r/LocalLLaMA/comments/1caw3ad/sharing_llama38bweb_an_action_model_designed_for/)。WebLlama 项目旨在推进基于 Llama 的智能 agent 开发。共享了[使用 OpenAI TTS 和 Whisper 与 Llama 3 8B 进行语音聊天的演示](https://v.redd.it/xwr67vtxkzvc1)。
*   **微调和扩展**：介绍了QDoRA用于[Llama 3模型的内存高效且准确微调](https://www.reddit.com/r/LocalLLaMA/comments/1cas7wg/qdora_efficient_finetuning_of_llama_3_with_fsdp/)，优于QLoRA和Llama 2。分享了[Hugging Face Space用于创建Llama 3模型的GGUF量化](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/)。讨论了[在微调Llama 3时添加BOS token的重要性](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/)。

**Llama 3 Performance and Capabilities**

*   **跟随指令**: 在 /r/LocalLLaMA 上, Llama-3-70B 因其[**跟随格式指令和提供简明扼要回复的能力**](https://www.reddit.com/r/LocalLLaMA/comments/1canrjq/llama370b_is_insanely_good_at_following_format/)而备受赞誉,没有多余的套话文本。
* 在/r/LocalLLaMA中分享了20个Llama 3 Instruct模型版本在不同量化级别下在HF、GGUF和EXL2格式之间的深入比较。主要发现包括[**EXL2 4.5bpw和GGUF 8位到4位量化表现出色**](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/)，而1位量化显示出了明显的质量下降。
* Groq托管的Llama-3-70B模型在解决侧向思维难题时，与HuggingChat版本相比存在一些差距，如 /r/LocalLLaMA 中所报告。[**温度设置会显著影响推理性能**](https://www.reddit.com/r/LocalLLaMA/comments/1casosh/groq_hosted_llama370b_is_not_smart_probably/)，0.4 设置可提供最佳一致性。

**Phi-3 and Llama 3 Models Push Boundaries of Open-Source Language AI**

*   **Phi-3 模型发布 3.8B、7B 和 14B 模型**: 在 /r/singularity 上, 发布了 Phi-3 模型, 该模型基于[**经过大量过滤的网络数据和合成数据**](https://www.reddit.com/r/singularity/comments/1cau7ek/phi3_released_medium_14b_claiming_78_on_mmlu/)进行训练。14B 模型在 MMLU 上的得分声称为 78%, 与 Llama 3 8B 相媲美, 尽管模型更小。权重将很快在 Hugging Face 上提供。

*   **Phi-3 3.8B接近GPT-3.5性能表现**：在/r/singularity上，Phi-3 3.8B模型[**在基准测试中接近GPT-3.5的性能表现**](https://www.reddit.com/r/singularity/comments/1cau3gy/phi3_a_small_38b_model_nears_gpt35_on_major/)，同时也有7B和14B容量的版本可用。模型权重与演示视频一同发布,展现了模型效率方面令人叹为观止的进步。

*   **Llama 3 70B在LMSYS排行榜上与GPT-4打平**: 在/r/singularity上, Llama 3 70B[**在LMSYS英语排行榜上获得了并列第一的成绩,与GPT-4-Turbo并列**](https://www.reddit.com/r/singularity/comments/1cau6yz/llama_3_70b_takes_second_place_in_the_english/)。它可以通过Groq API或Hugging Face免费使用。有人提出了关于竞技场排名有效性的质疑。

*   **Phi-3技术报告展现出色的基准测试结果**: 在/r/singularity中, Phi-3技术报告发布显示, [**3.8B模型可与Mixtral 8x7B相媲美, 在MMLU上达到69%, MT-bench为8.38**](https://www.reddit.com/r/singularity/comments/1catcdv/phi3_technical_report_impressive/)。7B和14B模型进一步扩展至MMLU 75%和78%。

*   **Llama 3参数翻倍带来递减收益**: 在/r/singularity上, 一张图表显示[**对同一数据集翻倍参数, 平均会将MMLU分数提高17%, 但对于Llama 3模型仅提高5%**](https://www.reddit.com/r/LocalLLaMA/comments/1caneis/doubling_the_parameters_on_the_same_dataset/), 这表明Llama 3已经高度优化。

**Miscellaneous**

* SambaNova Systems 使用 8 个 FP16 精度的芯片，展示了 [**每秒进行 430 个 token 的 Llama 3 8B 高速推理**](https://www.reddit.com/r/LocalLLaMA/comments/1caxbx6/sambanova_systems_running_llama_3_8b_at_430_tps/)。
*   **量化民主化**: 在 /r/LocalLLaMA 中引入了一个Hugging Face Space，[**旨在民主化创建Llama 3模型的GGUF量化**](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/)，从而提高了可靠性和可访问性。

* * *

AI Twitter Recap
================


**Perplexity AI Raises 62.7M at 1.04B Valuation**

*   **资金详情**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782784338238873769)和[@perplexity_ai](https://twitter.com/perplexity_ai/status/1782782211399279076)宣布,Perplexity AI在B1轮融资中筹集到**6270万美元**,估值达**10.4亿美元**,由**Daniel Gross**牵头,投资者包括Stan Druckenmiller、NVIDIA、Jeff Bezos、Tobi Lutke、Garry Tan、Andrej Karpathy、Dylan Field、Elad Gil、Nat Friedman、IVP、NEA、Jakob Uszkoreit、Naval Ravikant、Brad Gerstner和Lip-Bu Tan。
* 自2024年1月以来,Perplexity的服务规模持续增长,现已达到每月**169M个查询**,过去15个月内累计超过**10亿个查询**。Perplexity与**德国电信和软银**建立了合作关系,分发服务覆盖全球约**116M用户**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1782785026524135848)

*   **Perplexity 企业级 Pro 版本发布**: Perplexity 正在推出 **Perplexity 企业级 Pro** 版本,它提供了 **SOC2 合规性、单一登录 (SSO)、用户管理、企业级数据保留和安全警报**,以解决企业使用中的数据和安全问题。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1782775219733844256), [@perplexity\_ai](https://twitter.com/perplexity_ai/status/1782774382399557633)

**Meta's Llama-3 Model Achieves Top Performance**

*   **Llama-3性能表现**: Meta的**Llama-3 70B**模型已经跻身**Arena排行榜前5位**，超越了许多更大规模的模型。其8B版本也已经超越了许多更大的模型。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483699449332144)
*   **训练细节**: Llama-3 模型在**超过 15T 个 token 的数据**上进行了训练,并使用**SFT、拒绝采样、DPO 和 PPO**进行了对齐。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)
*   **英语性能**: Llama-3 70B在英语领域表现**更加出色**，评分约为**与GPT-4 Turbo并列第一**。它在人类偏好方面一直表现优异,始终与顶级模型平分秋色。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)

**Microsoft Releases Phi-3 Language Models**

*   **Phi-3 模型详细信息**: Microsoft 发布了 3 种尺寸的 **Phi-3** 语言模型: **phi-3-mini (3.8B)、phi-3-medium (14B) 和 phi-3 (7B)**。虽然尺寸很小,但 phi-3-mini **可与 Mixtral 8x7B 和 GPT-3.5 媲美**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655)
*   **训练数据**: Phi-3 模型经过使用"**经过大量过滤的网络数据和合成数据**"训练，所使用的数据量为 **3.3T tokens (mini) 和 4.8T tokens (small/medium)**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782598582731702764)
* 基准性能：Phi-3-mini在MMLU上达到了**68.8分**，在MT-bench上达到了**8.38分**。Phi-3-medium在MMLU上达到了**78%**，在MT-bench上达到了**8.9分**，超越了GPT-3.5。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655), [@\_akhaliq](https://twitter.com/_akhaliq/status/1782598582731702764)
*   Phi-3-mini **权重已根据MIT许可在Hugging Face上发布**。它经过优化可用于Hugging Face文本生成推理。[@_philschmid](https://twitter.com/_philschmid/status/1782781516172431685)

**Google's Gemini 1.5 Pro Achieves Strong Performance**

*   **Gemini 1.5 Pro 性能表现**：Google 的 **Gemini 1.5 Pro API** 现已荣登 **排行榜第2位**，超越 GPT-4-0125 接近榜首。在处理更长提示时，其性能更加出色，与 **GPT-4 Turbo** 并列 **第1名**。[@lmsysorg](https://twitter.com/lmsysorg/status/1782594507957223720)

**Other Notable Releases and Benchmarks**

* ByteDance发布了一款名为**Hyper-SD**的新型框架,该框架可用于图像生成领域中的多概念定制,并能在仅1-8个推理步骤内即实现SOTA性能水平。[@\_akhaliq](https://twitter.com/_akhaliq/status/1782601752417575423)
* JPMorgan 推出了 **FlowMind**，该系统利用 GPT 自动生成用于机器人流程自动化(RPA)任务的工作流程。[@\_akhaliq](https://twitter.com/_akhaliq/status/1782604054805332258)
* OpenAI提出了一个**指令层次**架构,以使大型语言模型(LLM)优先处理特权指令,并对提示注入和监狱突破更加稳健。[@_akhaliq](https://twitter.com/_akhaliq/status/1782607669376761989)

* * *

AI Discord Recap
================


**1\. Evaluating and Comparing Large Language Models**

* 围绕新发布的 **[Phi-3](https://arxiv.org/abs/2404.14219)** 和 **[LLaMA 3](https://llama.meta.com/llama3/)** 模型的性能和基准测试展开了讨论。有人对 **Phi-3** 的评估方法及其在 MMLU 等基准上可能存在的过拟合现象表示怀疑。

* 在各种任务上,**Phi-3**、**LLaMA 3**、**GPT-3.5**以及类似**Mixtral**的模型进行了比较。其中, **[Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** (3.8B)相比其尺寸显示出了令人瞩目的性能表现。

* 围绕评估真实模型能力的基准测试指标如**MMLU**、**BIGBench**和**LMSYS**的有效性和实用性存在争议,有意见认为随着模型不断进步,这些基准测试可能会变得不太可靠。

* 对开源版本 **Phi-3** 在 **MIT 许可证**下发布的期待，以及其承诺的多语言功能。

**2\. Advancements in Retrieval-Augmented Generation (RAG)**

* LlamaIndex 推出了[DREAM](https://twitter.com/llama_index/status/1781725652447879672)，这是一个用于尝试分布式 RAG（Retrieval-Augmented Generation）的框架,旨在构建健壮且可用于生产环境的 RAG 系统。

* 对于创新的 RAG 技术的探讨,包括用于高效的长上下文处理的 **[Superposition Prompting](https://arxiv.org/abs/2404.06910)**、用于提高检索质量的 **[CRAG](https://twitter.com/llama_index/status/1782799757376963006)**,以及带有函数调用的 **[RAG](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)**。

* 分享资源涉及 **[RAG evolution](https://arxiv.org/abs/2404.10981)**、**[credibility-aware generation](https://arxiv.org/abs/2404.06809)** 以及将检索与 LLM 规划集成以生成结构化输出等相关内容。

* 由 **[@JinaAI\_](https://twitter.com/llama_index/status/1782531355970240955)** 发布的开源 Reranker 版本可通过改进的向量搜索排名提升 RAG 性能。

**3\. Fine-tuning and Optimizing Large Language Models**

* 对于LLaMA 3的微调策略进行了广泛探讨,采用了如**Unsloth**等工具,解决了诸如Tokenizer配置、高效合并LoRA Adapter及注入知识等问题。

* 全面微调、**QLoRA**和**LoRA**方法的比较,**[QLoRA研究](https://twitter.com/teortaxesTex/status/1781963108036088060)** 表明其相较于LoRA可能具有更高的效率。

* 针对 **llm.c** 实现 **BF16/FP16** 混合精度训练, 相比 FP32 可取得约 **1.86倍性能提升**，详情见 **[PR #218](https://github.com/karpathy/llm.c/pull/218)**。

* 对于 **llm.c** 中的优化,包括利用 **CUDA** 内核改进（如 **GELU**、**AdamW**）以及采用 **thread coarsening** 等技术,以提升受内存限制的内核性能。

**4\. Multimodal and Vision Model Developments**

*   **[Blink](https://arxiv.org/abs/2404.12390)** 是一种新的基准测试,用于评估像 **GPT-4V** 和 **Gemini** 等多模态大型语言模型的核心视觉感知能力。

* 像[HiDiffusion](https://hidiffusion.github.io/) 这样的算法声称可以通过一行代码提高扩散模型的分辨率,也有通过流集成进行图像上采样的[PeRFlow](https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md)。

* 一款跨模态基础模型[SEED-X](https://arxiv.org/abs/2404.14396)，可以理解和生成任意尺寸的图像, 从而弥合现实世界应用程序中的差距。

* 基于 [Mixture-of-Attention (MoA)](https://snap-research.github.io/mixture-of-attention/) 架构的技术取得了进步, 可从语言中生成具有不同风格和个性化特点的图像。

**5\. Misc**

*   **Hugging Face平台停机导致模型访问中断**: 多家渠道报告在尝试使用**Hugging Face**时出现**504 Gateway Time-outs**和服务中断,影响了像**[LM Studio](https://x.com/lmstudioai/status/1782390856986550384)**等工具中的模型搜索和下载功能。有人推测Hugging Face可能采取了限流措施,正在进行消除对其依赖的长期解决方案。

* * *

