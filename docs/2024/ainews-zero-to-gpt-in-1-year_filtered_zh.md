Zero to GPT in 1 Year
================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-zero-to-gpt-in-1-year/](https://buttondown.email/ainews/archive/ainews-zero-to-gpt-in-1-year/) 

正如许多人所料,4月发布的GPT4T再次登顶[LMsys](https://twitter.com/lmsysorg/status/1778555678174663100),现已在[付费版ChatGPT](https://twitter.com/gdb/status/1778577748421644459)和一个全新的轻量级[模型评估Repo](https://x.com/swyx/status/1778589547200381425)中推出。我们[此前曾提到](https://www.latent.space/p/feb-2024),OpenAI将需要优先在ChatGPT中推出新模型,以助推增长。

总的来说，这是即将到来的Llama 3发布会前的一段宁静时刻。你可以查看[Elicit的论文/播客](https://x.com/swyx/status/1778520821386121582)或[Devin vs OpenDevin vs SWE-Agent的直播](https://twitter.com/hackgoofer/status/1778687452921888897)。不过,我们今天要特别提一下[**Vik Paruchuri**](https://twitter.com/VikParuchuri/status/1778534123138912366),他撰写了一篇文章,描述了[自己从工程师到在1年内打造出出色的OCR/PDF数据模型的历程](https://www.vikas.sh/post/how-i-got-into-deep-learning)。

![image.png](https://assets.buttondown.email/images/d7aa6b32-0769-4ca5-aafd-8801bd7a5d66.png?w=960&fit=max)

* * *


AI Twitter Recap
================


**GPT-4 and Claude Updates**

* GPT-4 Turbo重新登顶排行榜榜首：[@lmsysorg](https://twitter.com/lmsysorg/status/1778555678174663100)指出，GPT-4 Turbo在编程、更长查询和多语言能力等多个领域的表现均优于其他模型,已重新夺回了Arena排行榜第一名的位置。在仅包含英文提示和代码片段的对话中,它的表现更加出色。
*   **发布全新的 GPT-4 Turbo 模型**：[@sama](https://twitter.com/sama/status/1778578689984270543) 和 [@gdb](https://twitter.com/gdb/status/1778577748421644459) 宣布在 ChatGPT 中推出了一款全新的 GPT-4 Turbo 模型,该模型大幅提升了智能性和使用体验。[@miramurati](https://twitter.com/miramurati/status/1778582115460043075) 确认这是最新版的 GPT-4 Turbo。
* 新 GPT-4 Turbo 的评估数据: [@polynoamial](https://twitter.com/polynoamial/status/1778584064343388179) 和 [@owencm](https://twitter.com/owencm/status/1778619341833121902) 分享了评估结果, 与上一版本相比, MATH 提高了 +8.9%, GPQA 提高了 +7.9%, MGSM 提高了 +4.5%, DROP 提高了 +4.5%, MMLU 提高了 +1.3%, HumanEval 提高了 +1.6%。
* **Claude Opus仍优于新GPT-4**：[@abacaj](https://twitter.com/abacaj/status/1778435698795622516)和[@mbusigin](https://twitter.com/mbusigin/status/1778813997246034254)指出，在他们的使用中，Claude Opus较新GPT-4 Turbo模型更加智能、创造力更强。

**Open-Source Models and Frameworks**

*   **Mistral模型**：[@MistralAI](https://twitter.com/MistralAI)发布了新的开源模型，包括Mixtral-8x22B基础模型，这是一款很适合进行微调的强大模型([@\_lewtun](https://twitter.com/_lewtun/status/1778429536264188214))，以及Zephyr 141B模型([@osanseviero](https://twitter.com/osanseviero/status/1778430866718421198), [@osanseviero](https://twitter.com/osanseviero/status/1778816205727424884))。
* Medical mT5模型: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778607598784135261)分享了一个面向医疗领域的开源多语言文本到文本的大型语言模型 Medical mT5。
*   **LangChain 与 Hugging Face 集成**: [@LangChainAI](https://twitter.com/LangChainAI/status/1778465775034249625) 发布了跨模型提供商支持工具调用的更新,并提供了一个标准的 `bind_tools` 方法用于将工具附加到模型。 [@LangChainAI](https://twitter.com/LangChainAI/status/1778533665645134280) 还更新了 LangSmith,以支持在各种模型的跟踪中渲染工具和工具调用。
* Hugging Face Transformer.js：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1778456263161971172)指出，Transformer.js - 一个在浏览器中运行Transformer模型的框架 - 出现在Hacker News上。

**Research and Techniques**

* 从 Words 到 Numbers - LLMs 作为回归器: [@_akhaliq](https://twitter.com/_akhaliq/status/1778592009067925649) 分享了研究结果,分析了预训练的 LLMs 在给定上下文示例时,能够进行线性和非线性回归,并且可以媲美或优于传统的监督学习方法。
* **高效无限上下文Transformers**: [@_akhaliq](https://twitter.com/_akhaliq/status/1778605019362632077)分享了来自谷歌的一篇论文,介绍了如何将压缩记忆整合到常规注意力层中,使Transformer大语言模型能够以有限的内存和计算处理无限长的输入。
*   **OSWorld基准测试**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778599140634599721) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1778605020444795284) 分享了 OSWorld，这是首个可扩展的真实计算机环境基准测试，适用于多模态智能体，支持任务设置、基于执行的评估以及在各种操作系统上的交互式学习。
* **ControlNet++**: [@_akhaliq](https://twitter.com/_akhaliq/status/1778606395014676821) 分享了 ControlNet++，该模型通过高效的一致性反馈手段改善了扩散模型中的条件控制能力。
* **Applying Guidance in Limited Interval**：[@_akhaliq](https://twitter.com/_akhaliq/status/1778607531998232926)分享了一篇论文,表明在有限区间内应用引导能够提升扩散模型中的样本和分布质量。

**Industry News and Opinions**

*   **WhatsApp 与 iMessage 论辩**: [@ylecun](https://twitter.com/ylecun/status/1778745216842760502)将 WhatsApp 与 iMessage 的论辩比作公制与英制系统的争论,指出除了一些狂热使用 iPhone 的美国人或被禁止使用的国家外,全世界都在使用 WhatsApp。
*   **AI Agent将广泛应用**: [@bindureddy](https://twitter.com/bindureddy/status/1778508892382884265)预测 AI 代理将无处不在,使用 Abacus AI,您可以在短短 5 分钟到几小时内搭建这些Agent。
*   **Cohere Rerank 3 model**: [@cohere](https://twitter.com/cohere/status/1778417650432971225)和[@aidangomez](https://twitter.com/aidangomez/status/1778416325628424339)推出了Rerank 3，这是一个用于增强企业搜索和RAG系统的基础模型。该模型能够在100多种语言中准确检索多方面和半结构化数据。
* OpenAI 因信息泄露解雇员工: [@bindureddy](https://twitter.com/bindureddy/status/1778546797331521581) 报告称 OpenAI 解雇了 2 名员工,其中 1 人是 Ilya Sutskever 的亲密朋友,原因是泄露了一个内部项目的信息,很可能与 GPT-4 有关。


* * *

AI Discord Recap
================


*   **Mixtral 和 Mistral 模型正在获得关注**：**Mixtral-8x22B** 和 **Mistral-22B-v0.1** 模型正在引起关注,后者标志着首次成功将 Mixture of Experts (MoE) 模型转换为密集格式。讨论集中在它们的功能上,如 Mistral-22B-v0.1 拥有 22 亿参数。最近发布的 **[Zephyr 141B-A35B](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1)**，是 Mixtral-8x22B 的 Fine-tuned 版本。

*   **Rerank 3和Cohere的搜索增强功能**：**[Rerank 3](https://txt.cohere.com/rerank-3/)**，这是Cohere针对企业搜索和RAG系统推出的新型基础模型，支持100多种语言，拥有4k的上下文长度，并提供高达3倍的推理速度。该模型原生集成了**Elastic的推理API**以支持企业搜索。

* CUDA优化和量化探索：工程师优化CUDA库如CublasLinear, 以实现更快的模型推理,同时也在探讨4位、8位以及全新的高质量量化(HQQ)等量化策略。通过修改NVIDIA驱动程序, 可实现[在4090 GPU上开启P2P支持](https://x.com/__tinygrad__/status/1778676746378002712),从而显著提升速度。

* 一篇新论文《["Scaling Laws for Data Filtering"](https://arxiv.org/abs/2404.07177)》认为，数据整理工作不能忽视计算因素，并提出了处理非均匀网络数据的扩展法则。该研究社区正在探讨其含义并尝试理解所采用的实证方法。

其他值得关注的讨论包括:

* 推出了 **GPT-4 Turbo** 及其在编码和推理任务中的性能表现
* Ella的动漫图像生成能力欠佳
* 对于 Stable Diffusion 3 及其在解决当前模型限制方面的潜力的期待
* Hugging Face的Rerank模型下载量已达23万,且parler-tts库已正式发布。
* 围绕 OpenAI API、Wolfram 集成和 prompt 工程资源的讨论

* * *

