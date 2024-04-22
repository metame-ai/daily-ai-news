Mixtral 8x22B 引起cost efficiency讨论
=========================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/](https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/) 

Mistral继续通过[https://mistral.ai/news/mixtral-8x22b/]发布了一篇博文,以及他们8x22B模型的 Instruct-Tuned 版本。

![image.png](https://assets.buttondown.email/images/323db65b-608d-445d-83eb-1d6d9ce35e3f.png?w=960&fit=max)

这张图最终引发了[Databricks、Google和AI21](https://twitter.com/AlbertQJiang/status/1780648008696091003)之间的一些友好竞争, 体现了Mixtral在激活参数和MMLU性能之间创造了一个新的权衡。

![image.png](https://assets.buttondown.email/images/9677f3b7-64ba-4f12-af15-291dfda26c7d.png?w=960&fit=max)

事实上,激活参数数量并非与运行密集型模型的成本呈线性关系,而单一关注于MMLU对较缺乏严谨性的竞争对手来说并非理想选择。

* * *


AI Reddit Recap
===============


**AI Investments & Advancements**

* 来自科技巨头的大规模AI投资：在/r/singularity上，DeepMind CEO透露Google计划投资**超过1000亿美元**在AI领域，而微软、英特尔、软银以及阿布扎比基金等其他科技巨头也作出了类似的巨额投资[，这表明他们高度确信AI的发展潜力](https://www.bloomberg.com/news/articles/2024-04-16/deepmind-ceo-says-google-will-spend-more-than-100-billion-on-ai)。

* 英国将未经同意制作色情类Deepfake图像定为犯罪行为。在/r/technology讨论区,评论者[就其含义和执法挑战进行讨论](https://time.com/6967243/uk-criminalize-sexual-explicit-deepfake-images-ai/)。

*   **Nvidia的AI芯片主导地位**：在 /r/hardware 上，一名前Nvidia员工在Twitter上声称[本十年内将没有人追赶上Nvidia的AI芯片领先优势](https://i.redd.it/m388weqd9yuc1.png)，引发了关于该公司强大地位的讨论。

**AI Assistants & Applications**

*   **AI 伴侣潜在数十亿美元市场**：在 /r/singularity 上,一位科技高管预测 AI 女友可能成为一个**10亿美元的业务**。评论人士认为这种估计严重低估了实际情况,并[讨论了这种做法对社会的影响](https://www.yahoo.com/tech/tech-exec-predicts-ai-girlfriends-181938674.html)。

* [语言模型的无限上下文长度](https://twitter.com/_akhaliq/status/1780083267888107546)是 AI 语言模型的一个重大进步。这一新进展在 /r/artificial 上的一条推文中公布。

* 在 /r/artificial 中,一篇《自然》杂志的文章报道称[AI在几项基本任务中已经超越了人类performance](https://www.nature.com/articles/d41586-024-01087-4),但在更复杂的任务上仍然落后。

**AI Models & Architectures**

* 在 /r/LocalLLaMA 中，Zyphra 发布了 Zamba，这是一种7B参数的混合架构，结合了 Mamba 块和共享注意力机制。尽管训练数据较少，但它[的性能优于 LLaMA-2 7B 和 OLMo-7B 等模型](https://www.reddit.com/r/LocalLLaMA/comments/1c61k7v/zamba_a_7b_mambalike_ssm_hybrid_model_trained_for/)。该模型由一个7人团队在30天内使用128个 H100 GPU 开发而成。

* * *

AI Twitter Recap
================


**Mixtral 8x22B Instruct Model Release**

*   **出色的性能表现**: [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1780602023203029351)宣布发布Mixtral 8x22B Instruct模型,该模型在推理过程中仅使用**39B个激活参数**,就已显著超越现有的开放模型,速度快于70B模型。
*   **多语言能力**: [@osanseviero](https://twitter.com/osanseviero/status/1780595541711454602)指出 Mixtral 8x22B 精通**5种语言**（英语、法语、意大利语、德语、西班牙语），拥有**数学和编程能力**，且具有**64k的上下文窗口**。
*   该模型在 [@huggingface](https://twitter.com/huggingface) Hub 上以 **Apache 2.0 license** 授权提供, 可供下载并在本地运行, 这一点 [@_philschmid](https://twitter.com/_philschmid/status/1780598146470379880) 已经确认。

**RAG (Retrieval-Augmented Generation) Advancements**

*   **为提高精度而设计的GroundX**: [@svpino](https://twitter.com/svpino/status/1780571442096087224)分享称, @eyelevelai 发布了先进的RAG API GroundX。在对1,000页税务文件的测试中, **GroundX的准确率达到了98%**, 而LangChain和LlamaIndex分别为64%和45%。
*   **评估风险的重要性**：[@omarsar0](https://twitter.com/omarsar0/status/1780613738585903182)基于一篇关于 RAG 模型忠实度的论文,强调在使用可能包含支持性、矛盾或不正确数据的上下文信息的 LLM 时,有必要进行风险评估。
* **LangChain RAG 教程**: [@LangChainAI](https://twitter.com/LangChainAI/status/1780629875533181271) 在 @freeCodeCamp 上发布了一个讲解 RAG 基础知识和高级方法的播放列表。他们还分享了一个在使用 Mixtral 8x22B 进行 RAG 的 [@llama_index](https://twitter.com/llama_index/status/1780646484712788085) 教程。

**Snowflake Arctic Embed Models**


* **强大的嵌入模型**：[@SnowflakeDB](https://twitter.com/SnowflakeDB)在[@huggingface](https://twitter.com/huggingface)上开源了其Arctic嵌入模型系列，这是结合 @Neeva的搜索专业知识和Snowflake的AI能力的成果，正如[@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1780225794402627946)所述。
*   **效率和性能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1780621521230111181)强调了这些模型的高效性,**参数数量介于23M至335M之间**,**序列长度在512至8192个字符之间**,支持最多2048个token(无RPE)或8192个token(使用RPE)。
*   **LangChain 集成**: [@LangChainAI](https://twitter.com/LangChainAI/status/1780650806896947547)宣布同日支持将 Snowflake Arctic Embed 模型与其 @huggingface Embeddings 连接器一同使用。

**Misc**

*   **CodeQwen1.5 发布**: [@huybery](https://twitter.com/huybery/status/1780264890298720570) 介绍了 CodeQwen1.5-7B 和 CodeQwen1.5-7B-Chat 这些专门的 codeLLM 模型。它们使用了 **3T tokens** 的代码数据进行预训练, 展现出了出色的代码生成、长上下文建模(64K)、代码编辑和 SQL 能力, 在 SWE-Bench 中超越了 ChatGPT-3.5。
* 波士顿动力的新 Robot：[@DrJimFan](https://twitter.com/DrJimFan/status/1780622682561929645)分享了波士顿动力的新 Robot 视频，他认为人形 Robot 在未来10年内将超过 iPhone 的供给量，并且"人类水平"仅是一个人为的上限。
* 从第一天就拥有超级人工智能：[@ylecun](https://twitter.com/ylecun/status/1780596362415063217)表示,AI助手需要从一开始就具备类人的智能以及超人的能力,包括对物理世界的理解、持久内存、推理和分层规划。

* * *

AI Discord Recap
================


**Stable Diffusion 3 and Stable Diffusion 3 Turbo Launches**:

*   **Stability AI**推出了**Stable Diffusion 3**及其更快速的版本**Stable Diffusion 3 Turbo**,声称性能优于DALL-E 3和Midjourney v6。这些模型采用了新的**Multimodal Diffusion Transformer (MMDiT)**架构。
* 计划发布SD3权重供自托管使用,同时保持Stability AI开放式生成AI的方法,需有Stability AI会员资格。
* 社区正在等待有关SD3个人与商业使用的许可说明的进一步澄清。

**Unsloth AI Developments**:

* 关于 GPT-4 作为对 GPT-3.5 的细微优化迭代版本, 以及 Mistral7B 出色的多语言处理能力的探讨。
* 关于在Apache 2.0下开源发布**Mixtral 8x22B**的兴奋,该模型擅长于多语言流畅度和长上下文窗口。
* 对 Unsloth AI 的文档贡献和捐赠支持其发展深感兴趣。

**WizardLM-2 Unveiling and Subsequent Takedown**:

* Microsoft宣布推出**WizardLM-2**系列模型,包括8x22B、70B和7B等不同规模的型号,展现出了出色的性能表现。
* 然而，**WizardLM-2** 由于缺乏合规审查而未发布, 并非最初推测的毒性问题所致。
* 关于删除引起的困惑和讨论，部分用户表示对获取原始版本感兴趣。

*   **Stable Diffusion 3 推出并性能更出色**: **Stability AI** 发布了 **Stable Diffusion 3** 和 **Stable Diffusion 3 Turbo**,现已在其[开发者平台 API](https://bit.ly/3xHrtjG)上提供,拥有最快速可靠的性能。社区期待获得有关自托管 SD3 权重的 **Stability AI 会员** 模式的更多信息。同时, **SDXL 细节优化** 已使 SDXL refiner 几乎过时,用户讨论 **ComfyUI** 中的模型融合挑战以及 **diffusers** 管道的局限性。

* **WizardLM-2 正式发布引发兴奋和不确定性**：微软发布的 **WizardLM-2** 模型引发了人们对其可能拥有与 **GPT-4 类似功能**的开源格式的热情。然而,由于疏忽的合规审查导致模型突然被下架,这引发了困惑和猜测。用户们对比了 WizardLM-2 不同变体的性能,并分享了在 **LM Studio** 中解决兼容性问题的技巧。

*   **多模态模型在 Idefics2 和 Reka Core 中取得进步**：**Hugging Face 的 Idefics2 8B** 和 **Reka Core** 已经成为强大的多模态语言模型,在视觉问答、文档检索和编码方面展现出令人印象深刻的能力。Idefics2 的即将推出的面向聊天的变体和 Reka Core 在与行业巨头的竞争中的出色表现引起了广泛关注。讨论还围绕着像 **JetMoE-8B** 这样的模型的成本效益,以及 **Snowflake 的 Arctic embed family** 用于文本嵌入的推出。

其他值得关注的topic包括:

* 介绍了**ALERT**，这是一个[用于评估大型语言模型安全性的基准测试](https://github.com/Babelscape/ALERT)。还讨论了围绕AI安全标准的相关话题。
* 基于视觉应用程序的 **Retrieval Augmented Generation (RAG)** 探索, 以及 **World-Sim** 中 AI 仿真的哲学启示。
* AI-人类协作平台如[Payman AI](https://www.paymanai.com/)的兴起,以及 AI 推理在 **Supabase** 的边缘功能中的集成。
* 对**Chinchilla缩放定律**的挑战以及研究社区就**状态空间模型**表达能力进行的讨论。
*   关于 PEFT 方法（如 Dora 和 RSLoRA）的进展, 以及使用专家混合 (MoE) 方法进行多语言模型扩展的努力。

* * *

