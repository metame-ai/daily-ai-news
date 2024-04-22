
> 翻译转载自: [https://nlp.elvissaravia.com/p/top-ml-papers-of-the-week-689](https://nlp.elvissaravia.com/p/top-ml-papers-of-the-week-689) 


![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbba77782-3a8b-45e7-a1a6-68621ea3e4e3_1497x900.png)

* * *
**1). Llama 3** - 一个包括 8B 和 70B 预训练及指令微调模型的 LLM 系列; Llama 3 8B 在表现上优于 Gemma 7B 和 Mistral 7B Instruct; Llama 3 70 在多方面均优于 Gemini Pro 1.5 和 Claude 3 Sonnet。([论文](https://ai.meta.com/blog/meta-llama-3/) | [推文](https://x.com/AIatMeta/status/1780997403979735440))

* * *
**2). Mixtral 8x22B** - 一款新型的开源稀疏专家混合模型，相比其他社区模型而言，它在 MMLU 任务上展现出最佳的性能/成本比。该模型在推理、知识检索、数学和编码方面均有出色表现。([论文](https://mistral.ai/news/mixtral-8x22b/) | [推文](https://x.com/MistralAILabs/status/1780596888473072029))

* * *
**3).Chinchilla Scaling: A replication attempt** - 尝试复现 Hoffmann 等(2022年)提出的计算最优缩放定律的第三个估计方法(即 Chinchilla Scaling)；发现"**reported的估计与前两种估计方法不一致，无法拟合提取的数据，并报告了令人难以置信的过于窄的置信区间。**"([论文](https://arxiv.org/abs/2404.10102) | [推文](https://x.com/tamaybes/status/1780639257389904013))

* * *
**4). How Faithful are RAG Models?** - 旨在量化RAG和大型语言模型(LLMs)内部先验之间的相互影响；针对GPT-4和其他LLMs在问答任务上的分析发现，提供正确的检索信息可修复大部分模型错误(准确率达94%)；当文档包含更多不正确的值且LLM的内部先验较弱时，LLM更容易背诵不正确的信息；当LLMs具有更强的内部先验时, 它们表现得更加稳定。([论文](https://arxiv.org/abs/2404.10198) | [推文](https://x.com/omarsar0/status/1780613738585903182))

* * *
**5). A Survey on Retrieval-Augmented Text Generation for LLMs** - 全面概述了RAG领域的发展历程和挑战;详细讨论了RAG系统的四大重要方面:预检索、检索、后检索和生成。([论文](https://arxiv.org/abs/2404.10981) | [推文](https://x.com/omarsar0/status/1780961995178594324))

* * *
**6). The Illusion of State in State-Space Models** - 探讨了状态空间模型 (SSMs) 的表达能力,发现其与 Transformer 类似,存在局限性,SSMs 不能表达复杂度为 TC^0 之外的计算;发现 SSMs 不能解决状态跟踪问题,如置换合成(permutation composition),以及评估代码或跟踪长篇叙事中实体等任务。([论文](https://arxiv.org/abs/2404.08819) | [推文](https://x.com/lambdaviking/status/1780246351520887281))

* * *
**7). Reducing Hallucination in Structured Outputs via RAG** - 讨论了如何部署高效的 RAG 系统用于结构化输出任务;RAG 系统将一个小型语言模型与一个极小的检索器相结合;它表明 RAG 可以在受限资源环境中部署强大的 LLM 驱动系统,同时缓解幻觉等问题,提高输出的可靠性。([论文](https://arxiv.org/abs/2404.08189) | [推文](https://x.com/omarsar0/status/1779896289745846778))

* * *
**8). Emerging AI Agent Architectures** - 简要概括了新兴的 AI Agent 架构；它将讨论重点放在了推理、规划和工具调用等功能上，这些都是构建复杂 AI 驱动代理工作流和系统所需的。该报告包含了当前的功能、局限性、洞见以及未来 AI Agent 设计发展的构想。([论文](https://arxiv.org/abs/2404.11584) | [推文](https://x.com/omarsar0/status/1780958785785200756))

* * *
**9). LLM In-Context Recall is Prompt Dependent** - 分析了不同 LLM 在几个针对性测试中的上下文召回表现；展示了不同 LLM 在不同长度和深度上对事实的召回能力；发现模型的召回表现受到提示的微小变化的显著影响；提示内容和训练数据之间的相互作用可能降低响应质量；通过增加规模、增强注意力机制、尝试不同的训练策略以及进行微调可以提高模型的召回能力。([论文](https://arxiv.org/abs/2404.08865) | [推文](https://x.com/omarsar0/status/1780244042007122129))

* * *
**10). A Survey on State Space Models** - 一篇关于状态空间模型(SSM)的调研论文,包含实验对比分析;文章回顾了当前的SSM技术,与替代方案相比的改进,面临的挑战以及它们的应用。([论文](https://arxiv.org/abs/2404.09516) | [推文](https://x.com/omarsar0/status/1781430319926686190))

* * *
