FineWeb: 15T Tokens, 12 years of CommonCrawl
==============================================================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/) 

尽管 Redpajama 2 [提供了多达 30T tokens](https://www.reddit.com/r/LocalLLaMA/comments/17om8xf/redpajamadatav2_is_incredible/)，但大多数 2023 年的大型语言模型(LLMs)仅使用了最多 2.5T tokens 进行训练 - 但随后 [DBRX 推出了 12T tokens](https://buttondown.email/ainews/archive/ainews-dbrx-best-open-model-but-not-most-efficient/)，Reka [Core/Flash/Edge 使用了 5T 个 tokens](https://twitter.com/RekaAILabs/status/1779894622334189592)，Llama 3 使用了 [15T 个 tokens](https://ai.meta.com/blog/meta-llama-3/)。如今 Hugging Face 已经发布了一个开放数据集,包含 12 年经过过滤和去重的 CommonCrawl 数据,总共有 15T tokens。

![image.png](https://assets.buttondown.email/images/399c8bc7-ff7b-4b85-824e-727b238db21c.png?w=960&fit=max)

值得注意的是,Guilherme曾经在[TII UAE Falcon 40B团队](https://x.com/ClementDelangue/status/1782065141200073122)工作,并负责他们的[RefinedWeb数据集](https://arxiv.org/abs/2306.01116)的相关工作。

在 Llama 3 发布一周后，如果您拥有所需的计算能力与代码，您现已可以训练属于自己的 Llama 3 模型。

* * *


AI Reddit Recap
===============


AI Models and Capabilities

* **WizardLM-2-8x22b性能表现**: 根据一位用户的基准测试结果,在/r/LocalLLaMA中,WizardLM-2-8x22b在推理、知识和数学测试方面[优于](https://www.reddit.com/r/LocalLLaMA/comments/1c9s4mf/wizardlm28x22b_seems_to_be_the_strongest_open_llm/)其他开放LLM,如Llama-3-70b-instruct。
* 在 /r/LocalLLaMA 中, Claude Opus 展示了使用 [0-shot prompting](https://www.reddit.com/r/LocalLLaMA/comments/1ca12yg/claude_opus_can_spot_this_error_in_my_code_with) 检测代码错误的出色能力, 在此任务上优于 Llama 3 和其他模型。
* Llama 3在/r/LocalLLaMA中展示了[令人印象深刻的zero-shot角色扮演能力](https://www.reddit.com/r/LocalLLaMA/comments/1c9v2k3/the_incredible_zeroshot_roleplay_ability_of_llama3/)。

Benchmarks and Leaderboards

*   **LMSYS 聊天机器人排行榜局限性**：在 /r/LocalLLaMA 上,人们提出了一些担忧,认为随着像 Llama 3 这样的经过指令微调的模型能够操纵基准测试,LMSYS 聊天机器人排行榜在评估真实模型性能方面变得越来越无用。需要更加全面的基准测试。
*   **新的 RAG 基准测试结果**：在 /r/LocalLLaMA 发布了一个[新的 RAG 基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1c9whsv/new_rag_benchmark_including_llama3_70b_and_8b/)，比较了 Llama 3、CommandR、Mistral 等在复杂商业文档问答任务上的表现。Llama 3 70B 的表现未能达到 GPT-4 水平。Mistral 8x7B 仍然是一个性能出色的全能模型。

Quantization and Performance

*   **高效的 Llama 3 量化模型**：/r/LocalLLaMA 注意到 [Huggingface 上 quantfactory 发布的 Llama 3 量化模型](https://www.reddit.com/r/LocalLLaMA/comments/1c9qufe/note_on_llama_3_quantized_models/)是目前最优秀的选择。
* **AQLM 对 Llama 3 8B 的量化**：[AQLM 对 Llama 3 8B 的量化](https://www.reddit.com/r/LocalLLaMA/comments/1c9uvlk/aqlm_quantization_for_llama38b/)已经显示可以在 Transformers 和 text-generation-webui 中加载,在初始测试中的性能与基线持平。

Censorship and Safety

*   **AI使用禁令适用于性犯罪者**: 据报道,一名英国性犯罪者因制作儿童不雅图像而被禁止使用AI工具,引发了慈善机构的担忧。这些慈善机构希望科技公司能够阻止此类内容的生成。
* **GPT-4 利用能力**: GPT-4可以[以87%的成功率读取安全公告来利用真实漏洞](https://www.reddit.com/r/OpenAI/comments/1c9mw4d/gpt4_can_exploit_real_vulnerabilities_by_reading/)，表现优于其他大语言模型和扫描器,引发了对未来更大语言模型可能使利用攻击更加容易的担忧。
*   **AI生成的不安全信息**：在 /r/LocalLLaMA 上有讨论,是否 [AIs 具备产生既不广为人知的独特不安全信息的能力](https://www.reddit.com/r/LocalLLaMA/comments/1c9n6ci/are_ais_actually_capable_of_producing_uniquely/)。大多数示例似乎都是基本概述,而非真正敏感的知识。


* * *

AI Twitter Recap
================


**Meta Llama 3 Release**

*   **模型详情**: [@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440)发布了Llama 3模型，其尺寸为**8B和70B**，同时还有一个**正在训练的400B+模型**。Llama 3使用了**128K词汇tokenizer**和**8K上下文窗口**。它是在**15T个token**上进行训练的，并采用**SFT、PPO和DPO**技术在10M个样本上进行了微调。
*   **性能**: [@karpathy](https://twitter.com/karpathy/status/1781028605709234613)指出,Llama 3 70B在MMLU等基准测试中正在**接近GPT-4水平的性能**。8B模型的性能超过了其他如Mistral 7B的模型。[@DrJimFan](https://twitter.com/DrJimFan/status/1781006672452038756)强调,它将是首个达到GPT-4水平的开源模型。
*   **计算和扩容**：[@karpathy](https://twitter.com/karpathy/status/1781387674978533427)估计 **1.3M A100小时用于8B模型，6.4M小时用于70B模型**，在一个24K GPU集群上有400 TFLOPS的吞吐。与计算最优化扩容比例相比，模型严重**欠训练**。
* 模型可通过 [@huggingface](https://twitter.com/huggingface)、[@togethercompute](https://twitter.com/togethercompute/status/1781004579817349266)、[@AWSCloud](https://twitter.com/awscloud)、[@GoogleCloud](https://twitter.com/GoogleCloud) 以及更多渠道获得。4位量化版本允许在消费级硬件上运行 **8B** 模型。

**Reactions and Implications**

*   **开源 AI 进展**：很多人认为这是**open-source AI 发展的一个重要时刻**,已经超越了封闭模型。[@bindureddy](https://twitter.com/bindureddy/status/1781028123313881206)和其他人预测,开源模型将在短短几周内**达到 GPT-4 级别的能力**。
* **LLM的商品化**: [@abacaj](https://twitter.com/abacaj/status/1781443464246559180)和其他人指出,随着人们优化运行时间和蒸馏,这将**促使成本下降**。有些人猜测这可能会挑战OpenAI的业务模式。

*   **微调和应用**: 许多人, 包括 [@maximelabonne](https://twitter.com/maximelabonne/status/1781248104479494581) 和 [@rishdotblog](https://twitter.com/rishdotblog/status/1781208858612138329), 已经在为编码、开放式问答等方面 **微调 Llama 3**。预计将出现 **强大的开放模型和应用程序的大量涌现**。

**Technical Discussions**

* **指令微调(Instruction Finetuning)**: [@Teknium1](https://twitter.com/Teknium1/status/1781345814633390579) 认为 Llama 3 的性能已经否定了近期提出的观点,即微调无法教会模型新的知识或功能。
*   **过度训练和缩放**：[@karpathy](https://twitter.com/karpathy/status/1781033433336262691) 及其他人探讨了**训练模型远超计算最优比例**可获得在推理上高效的强大模型的情况，这可能会改变最佳实践。
* Tokenizer和数据：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1781001629174575126)指出,改进的 **128K Tokenizer** 对于效率非常重要,尤其是对于多语言数据。高质量的训练数据是关键重点。

* * *

AI Discord Recap
================


*   **Llama 3**: Meta 发布的 **Llama 3** 引发了广泛讨论,70B 参数模型的性能与 GPT-4 不相伯仲([来自 Teknium 的推文](https://x.com/teknium1/status/1781328542367883765)),而 8B 版本的性能还超过了 Claude 2 和 Mistral。Unsloth AI 已整合了 Llama 3,承诺实现 **2 倍训练速度和 60% 内存使用降低**([GitHub 发布](https://github.com/unslothai/unsloth/releases/tag/April-Llama-3-2024))。一个[初学者指南视频](https://youtu.be/r-heqmMYNL0)解释了该模型的 Transformer 架构。

*   **Tokenizer问题和微调修复措施**: 微调**Llama 3**遇到了挑战,缺少BOS令牌导致训练期间出现高损失和 `grad_norm inf`。通过此[tokenizer配置PR](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41)提供了解决方案。该模型庞大的tokenizer词汇表引发了关于效率和必要性的讨论。

*   **推理速度突破**：**Llama 3** 在Groq Cloud上实现了**每秒800个tokens**([YouTube视频](https://www.youtube.com/watch)), Unsloth用户在7900XT等AMD GPU上报告了每秒最高60个tokens。讨论还强调了Llama 3 在Groq上70B模型的首字节响应时间小于100毫秒。

*   **评估和比较大型语言模型**: 对话比较了 **Llama 3** 与 **GPT-4**、**Claude** 及其他模型,尽管 Llama 3 70B 的 lmsys 评分不错,但仍无法完全匹敌 GPT-4 Turbo。发布 **FineWeb** 数据集 ([来自 Guilherme Penedo 的推文](https://x.com/gui_penedo/status/1781953413938557276)),拥有 15 万亿个词汇,这表明其有望超越现有的 RefinedWeb 和 The Pile 等数据集。

*   **新兴工具和框架**: 讨论了几种新的工具和框架,包括 Facebook Research 的 **Hydra** 用于配置复杂应用程序, **LiteLLM** ([网站](https://litellm.vercel.app/)) 作为 LLM 项目的模板, **Prompt Mixer** ([网站](https://www.promptmixer.dev/)) 用于协作式提示工程, 以及 **WhyHow.AI's Knowledge Graph SDK** ([Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)) 用于模式控制的自动知识图谱。

* 检索增强型生成(Retrieval-Augmented Generation, RAG)的进展：RAG 的发展是一个热门话题,提出了一个新的评估 RAG 模型的基准([Stella Biderman 的 Tweet](https://x.com/BlancheMinerva/status/1782437494585282965)),以及使用 Llama 3 构建 [RAG 聊天机器人的指南](https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3)和使用 LangChain 的自我查询检索器进行 [租赁公寓搜索的教程](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever)。

*   **强化学习从人类反馈(RLHF)洞见**：一篇新论文[从$r$到$Q^*$:你的Language Model是秘密的一个Q-Function](https://arxiv.org/abs/2404.12358)将传统的RLHF方法与直接偏好优化(DPO)进行了对比,并将理论与标准RLHF方法和贝尔曼方程相一致。

* 优化Transformer模型：讨论了优化Transformer模型的技术，包括在推理时[**通过**]近似注意力机制压缩token长度([arXiv:2401.03462](https://arxiv.org/abs/2401.03462)、[arXiv:2401.06104](https://arxiv.org/abs/2401.06104))、采用Activation Beacon和TOVA等方法[**以延长**]上下文长度，以及动态分配FLOPs ([arXiv:2404.02258](http://arxiv.org/abs/2404.02258))。

*   **伦理考虑和法律影响**：对话探讨了 AI "越狱"的伦理影响及其可能导致的非预期 Agent 行为,以及使用如 **Nightshade** 等可能与 **计算机欺诈与滥用法案 (CFAA)** 相冲突的工具所涉及的法律风险。

*   **协作努力与社区参与**：许多渠道促进了如 **minbpe-rs**（[GitHub](https://github.com/gnp/minbpe-rs)）—— minbpe 的 Rust 移植版本 ——以及使用 **Cohere Command R+** 的开源匹配 AI 应用程序（[推文](https://x.com/anmol_desai2005/status/1781679469679325605)）等项目的协作。社区成员还分享了学习资源，如[微调 LLM 的课程](https://github.com/andysingal/llm-course/blob/main/llama_finetune/Fine-tune-basics.md)和[Eugene Yan 关于评估 LLM 的博客文章](https://eugeneyan.com/writing/abstractive/)。

* * *

