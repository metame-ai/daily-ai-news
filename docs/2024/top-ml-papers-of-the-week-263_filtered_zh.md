
> 翻译转载自: [https://nlp.elvissaravia.com/p/top-ml-papers-of-the-week-263](https://nlp.elvissaravia.com/p/top-ml-papers-of-the-week-263) 


![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4ba24d4-ea0e-4dc8-b344-472ebaa3c47a_1497x901.png)

* * *
**1).Leave No Context Behind** - 将压缩式内存集成到标准点积注意力层中;目标是使Transformer语言模型能够有效处理无限长输入,同时具有有限的内存占用和计算量;提出了一种新的注意力技术称为Infini-attention,它将压缩式内存模块并入标准注意力机制;它将遮蔽局部注意力和长期线性注意力集成到单个Transformer模块中;这使得Infini-Transformer模型能够高效处理长短范围的上下文依赖;在长上下文语言建模任务中,以114倍的内存压缩比优于基线模型。([paper](https://arxiv.org/abs/2404.07143) | [tweet](https://x.com/omarsar0/status/1778480897198612839))

* * *
**2). OpenEQA** - 提出了一个开放词汇基准数据集,用于评估AI模型执行具象问答(Embodied Question Answering, EQA)的能力。该数据集包含来自180个真实环境的1600个人类生成的问题。同时提供了一种基于大型语言模型(LLM)的评估协议,结果显示类似GPT-4V的模型与人类水平存在较大差距。([paper](https://open-eqa.github.io/assets/pdfs/paper.pdf) | [tweet](https://x.com/AIatMeta/status/1778425321118732578))

* * *
**3). CodeGemma** - 基于Gemma的开放代码LLM家族；CodeGemma 7B模型在数学推理方面表现出色，与其他开源模型的代码能力相匹配；经过指令微调的CodeGemma 7B模型在通过HumanEval基准测试评估的Python编码方面更为强大；结果还表明，该模型在7B模型中在GSM8K上表现最佳；CodeGemma 2B模型实现了同步规模下最优的代码补全, 并针对延迟敏感环境中的快速代码补全和部署而设计。([paper](https://storage.googleapis.com/deepmind-media/gemma/codegemma_report.pdf) | [tweet](https://x.com/omarsar0/status/1777723836202467713))

* * *
**4). LM-Guided Chain-of-Thought** - 将知识蒸馏应用于小型 LM,并利用大型 LM 生成的推理来缩小推理能力差距;推理由轻量级 LM 生成,答案预测则由冻结的大型 LM;这种资源高效的方法避免了对大模型进行Fine-tune,而是将推理生成外包给小型语言模型;经过知识蒸馏的 LM 进一步利用多个以理性为导向和任务为导向的奖励信号进行强化学习优化;本文提出的基于 LM 的 CoT 提示方法优于标准提示和 CoT 提示。Self-consistency解码也提高了性能。([paper](https://arxiv.org/abs/2404.03414) | [tweet](https://x.com/omarsar0/status/1777755819150373121))

* * *
**5). Best Practices and Lessons on Synthetic Data** - 由Google DeepMind概述的合成数据研究,涵盖应用、挑战和未来方向;探讨在使用合成数据时需要注意的重要议题,如确保数据质量、真实性、保真度、无偏见性、可信度、隐私性等。([paper](https://arxiv.org/abs/2404.07503) | [tweet](https://x.com/omarsar0/status/1778804848038683066))

* * *
**6). Reasoning with Intermediate Revision and Search** - 提出了一种适用于可分解任务的通用推理和搜索方法;所提出的基于图的框架THOUGHTSCULPT融入了迭代自我修正能力,使LLM能够构建思维网络;与Tree-of-thoughts等方法不同,这种新方法采用蒙特卡罗树搜索(MCTS)高效地探索搜索空间;由于具有持续思维迭代的能力,THOUGHTSCULPT特别适用于开放式生成、多步推理和创意激发等任务。([paper](https://arxiv.org/abs/2404.05966) | [tweet](https://x.com/omarsar0/status/1777896810805186757))

* * *
**7). Overview of Multilingual LLMs** - 一篇关于多语言 LLMs 的调研报告, 包括对方法的深入评论、分类、新兴前沿、挑战以及推进研究的资源。([paper](https://arxiv.org/abs/2404.04925) | [tweet](https://x.com/omarsar0/status/1778063103906771105))

* * *
**8). The Physics of Language Models** - 研究知识容量的缩放规律,通过loss或基准评估模型的能力,来估计模型存储的知识位数; 报告称"_语言模型每个参数最多只能存储2比特知识,即使量化为int8,这种知识可以灵活地用于下游应用。因此,一个7B模型可以存储14B比特的知识,超过了英语维基百科和教科书的总和。_"([paper](https://arxiv.org/abs/2404.05405) | [tweet](https://x.com/omarsar0/status/1777709227319968034))

* * *
**9). Aligning LLMs to Quote from Pre-Training Data** - 提出了将大型语言模型(LLM)对齐以直接从预训练数据中引用记忆信息的技术。该对齐方法不仅能生成高质量的逐字引用语句,而且整体上保留了响应质量。该方法利用合成偏好数据集进行引用,无需任何人工标注,并使用偏好优化将目标模型对齐以实现引用。([paper](https://arxiv.org/abs/2404.03862) | [tweet](https://x.com/omarsar0/status/1777408054402646433))

* * *
**10). The Influence Between NLP and Other Fields** - 旨在量化23个研究领域与自然语言处理之间的影响程度；自然语言处理的跨领域参与度从1980年的0.58下降到2022年的0.31；研究还发现，自然语言处理的引用被计算机科学主导，占引用总数的80%以上，重点是人工智能、机器学习和信息检索；总的来说，自然语言处理变得更加封闭 -- 同一领域内的引用增长较快，跨学科研究有所下降。([paper](https://aclanthology.org/2023.emnlp-main.797/) | [tweet](https://x.com/omarsar0/status/1777337237794955586))
