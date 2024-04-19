Meta Llama 3 (8B, 70B)
=================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-to-be-named-5820/](https://buttondown.email/ainews/archive/ainews-to-be-named-5820/) 

今日最热新闻，[Meta部分发布了Llama 3](https://ai.meta.com/blog/meta-llama-3/)。今天，8B和70B版本问世,但最引人注目的是400B版本(仍在训练中),被广泛誉为首个达到GPT-4级别的开源模型。

![image.png](https://assets.buttondown.email/images/a004405a-73b2-4d6e-9eae-2a2d8cf8927b.png?w=960&fit=max)


* * *


AI Reddit Recap
===============


**Key Themes in Recent AI Developments**

*   **Stable Diffusion 3 发布和比较**：Stability AI 已经发布了[Stable Diffusion 3 API](https://stability.ai/news/stable-diffusion-3-api)，模型权重即将推出。[SD3 和 Midjourney V6](https://www.reddit.com/r/StableDiffusion/comments/1c6iae0/sd3_vs_midjourneyv6/) 的比较结果参差不齐，而[真实性测试彰显了 SD3 的能力](https://www.reddit.com/gallery/1c6un6f)。Emad Mostaque [确认 SD3 权重将与 ComfyUI 工作流程一同发布](https://i.redd.it/60wquhb3x2vc1.jpeg)在 Hugging Face 平台上。

* 机器人与AI智能体的进步：波士顿动力公司揭示了他们的人形机器人Atlas的[电动版本](https://www.youtube.com/watch)，拥有令人印象深刻的灵活性。[Menteebot是一款可通过自然语言控制的人形AI机器人](https://www.yahoo.com/tech/menteebot-is-a-human-sized-ai-robot-that-you-command-with-natural-language-110052927.html)。微软的[VASA-1模型能在RTX 4090上以每秒40帧的速度实时生成逼真的说话面部](https://www.microsoft.com/en-us/research/project/vasa-1/)。

*   **新语言模型和基准测试**: [Mistral，一家欧洲的OpenAI竞争对手，正在寻求50亿美元的融资](https://www.reuters.com/technology/frances-mistral-ai-seeks-funding-5-bln-valuation-information-reports-2024-04-17/)。他们的[Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)在64K上下文的基准测试中以100%的准确性超越了开源模型。新的[7B合并模型融合了不同基础的优势](https://huggingface.co/Noodlz)。[Coxcomb，一款7B创意写作模型](https://huggingface.co/N8Programs/Coxcomb)在基准测试中表现良好。

*   **AI 安全与监管讨论**：OpenAI 前董事会成员 Helen Toner [呼吁对顶尖 AI 公司进行审核](https://www.bloomberg.com/news/articles/2024-04-16/former-openai-board-member-calls-for-audits-of-leading-ai-companies)，以分享其能力和风险信息。[摩门教会发布了 AI 使用原则](https://newsroom.churchofjesuschrist.org/article/church-jesus-christ-artificial-intelligence)，指出了其好处和风险。


*   **用于AI开发的工具和框架**：[Ctrl-Adapter](https://v.redd.it/xugl158ya2vc1)框架可将控制适配到扩散模型。[Distilabel 1.0.0](https://github.com/argilla-io/distilabel)可利用LLM打造合成数据集流程。[Data Bonsai](https://github.com/databonsai/databonsai)使用LLM清理数据,并集成ML库。[Dendron](https://github.com/richardkelley/dendron)则利用行为树构建LLM智能体。


* * *

AI Twitter Recap
================


* * *

**Meta Llama 3 Release**

*   **Llama 3模型发布**: [@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440)宣布发布Llama 3 8B和70B模型,提供**优化的推理能力**,并达到了**同类模型的新SOTA水平**。未来几个月将有更多模型、功能和研究论文问世。
* 模型细节: [@omarsar0](https://twitter.com/omarsar0/status/1780992539891249466) 指出 Llama 3 采用了一个**标准的解码器型 Transformer**架构，使用了**128K token的词汇表**、**8K的序列长度**、**分组查询注意力机制**、**15T 个预训练令牌**，并采用了诸如**SFT、拒绝采样、PPO 和 DPO 等对齐技术**。
*   **性能**：[@DrJimFan](https://twitter.com/DrJimFan/status/1781006672452038756)对比了Llama 3 400B的性能与Claude 3 Opus、GPT-4和Gemini,发现其**正逼近GPT-4水平**。[@ylecun](https://twitter.com/ylecun/status/1780999981962342500)也强调了8B和70B模型在**基准测试中的出色表现**。

**Open Source LLM Developments**

*   **Mixtral 8x22B 发布**: [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1780602023203029351)发布了 Mixtral 8x22B，这是一个具有**141B个参数(39B个活跃参数)**、**多语言功能**、**本地函数调用**以及**64K的上下文窗口**的开放模型。它为开放模型树立了**新的标准**。
*   **Mixtral性能表现**: [@bindureddy](https://twitter.com/bindureddy/status/1780609164223627291)指出 Mixtral 8x22B具有**最佳的性价比**，拥有**出色的MMLU性能**并具备通过微调超越GPT-4的潜力。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1780605940842021327)强调了其**优秀的数学能力**。
*   **开放模型排行榜**：[@bindureddy](https://twitter.com/bindureddy/status/1780797091465527736)和[@osanseviero](https://twitter.com/osanseviero/status/1780717276771344895)分享了开放模型排行榜，展示了**开放模型的迅速进展和广泛传播**。Llama 3有望推动这一进程的进一步发展。

**AI Agents and RAG (Retrieval-Augmented Generation)**

*   **RAG基础知识**: [@LangChainAI](https://twitter.com/LangChainAI/status/1780629875533181271)与@RLanceMartin和@freeCodeCamp合作,发布了一个**讲解RAG基础知识和高级方法的视频合集**。
*   **Mistral RAG Agents**: [@llama_index](https://twitter.com/llama_index/status/1780646484712788085) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1780763995781378338) 分享了有关**使用 @MistralAI 的新 8x22B 模型构建 RAG agents 的教程**，展示了文档路由、相关性检查和工具使用等内容。
*   **RAG的忠实度**: [@omarsar0](https://twitter.com/omarsar0/status/1780613738585903182)发表了一篇论文, **定量分析了大语言模型内部知识与检索信息之间的矛盾**, 并突出了在信息敏感领域部署大语言模型的影响。

**AI Courses and Education**

*   **Google ML课程**: [@svpino](https://twitter.com/svpino/status/1780657510518788593)分享了**300小时的免费Google机器学习工程师课程**，涵盖从初级到高级水平。
*   **Hugging Face课程**:[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1780612212765200599)宣布推出**关于Hugging Face量化基础知识的新免费课程**,旨在使开放源码模型更加易于访问和高效。
*   **斯坦福大学 CS224N 学生统计信息**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1780640708497699186)分享了本学期 **CS224N 共有 615 名学生的统计信息**,显示了涵盖各专业和年级的广泛代表性。

**Miscellaneous**

*   **Zerve作为Jupyter的替代方案**: [@svpino](https://twitter.com/svpino/status/1780938523844968627)建议使用Zerve，这是一个**具有与Jupyter不同理念的基于Web的IDE**,可能会取代Jupyter notebook应用于许多使用场景。它具有**ML/DS工作流程的独特功能**。

* * *

AI Discord Recap
================


**Llama 3 Launch Generates Excitement**: 
* 承诺**改进推理能力**，并在多个任务中树立"新的最先进"基准。
* 可通过合作伙伴提供的[Together AI的API](https://together-ai.webflow.io/blog/together-ai-partners-with-meta-to-release-meta-llama-3-for-inference-and-fine-tuning)进行**推理和细调**, 最高可达 350 个 tokens/秒。
* 对于即将推出的**400B+参数版本**,人们正期待。

**Mixtral 8x22B Redefines Efficiency**: 
* 利用稀疏 Mixture-of-Experts (MoE) 架构，使用 39B 个活跃参数，总计 141B 个参数。
* 支持 **64K token 上下文窗口**以实现精准信息召回。
* 在Apache 2.0开源许可下发布,并配合Mistral的自定义Tokenizer。

**Tokenizers and Multilingual Capabilities Scrutinized**: 
* Llama 3 的 **128K 词汇表 tokenizer** 涵盖了 30 多种语言，但可能无法很好地处理非英语任务。
* Mistral开源了其**具有工具调用和结构化输出的Tokenizer**，以标准化微调。
* 关于**更大的 Tokenizer 词汇表有益于多语言 LLMs** 的讨论。

**Scaling Laws and Replication Challenges**: 

* [Chinchilla scaling paper](https://arxiv.org/abs/2404.10102)的发现遭到质疑,作者承认错误并开放数据源。
* 关于是否 **Scaling Laws** 是否成立分歧的观点。
* 基于有限数据外推时，呼吁采用更为 Realistic 的实验次数和更窄的置信区间。

**Misc**

*   **Llama 3 发布引发兴奋和审慎**：Meta 发布了具有 8B 和 70B 参数模型的 **[Llama 3](https://llama.meta.com/llama3/)**,在 AI 社区引发了广泛兴趣和测试。工程师对其性能超越前代 **Llama 2** 和 **GPT-4** 的表现赞叹不已,但也指出了 128k token 的上下文窗口等局限性。集成工作正在 **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536)** 和 **[Unsloth](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp)** 等框架中进行,量化版本正在 **[Hugging Face](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF)** 上出现。然而,一些人对 Llama 3 在下游使用方面的许可限制表示担忧。

* **Mixtral和WizardLM推动开源边界**: [Mistral AI的Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/)和微软的[WizardLM-2](https://wizardlm.github.io/WizardLM2)作为强大的开源模型引发了广泛关注。Mixtral 8x22B拥有39亿个活跃参数,在数学、编程和多语言任务方面表现出色。WizardLM-2提供了一个8x22B的旗舰版本和一个性能更出色的7B变体。这两款模型都展现了开源模型的快速进步,并获得了如[OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct)和[LlamaIndex](https://twitter.com/llama_index/status/1780646484712788085)等平台的支持。

* SD3反馈不一: Stability AI发布了[SD3的API](https://vxtwitter.com/StabilityAI/status/1780599024707596508),但初步反馈参差不齐。虽然在字体排印和提示词遵循方面有所进步,但部分用户反映存在性能问题和价格大幅上涨。该模型缺乏本地使用支持也引发批评,不过 Stability AI 承诺将很快向成员提供权重文件。

* CUDA难题与优化：CUDA工程师应对了各类挑战,涉及**[tiled matrix multiplication](https://discord.com/channels/1189498204333543425/1189498205101109300/1230259495330906194)**到**[自定义内核与torch.compile的兼容性](https://discord.com/channels/1189498204333543425/1189607750876008468/1230436794705903617)**。讨论深入探讨了内存访问模式、warp分配以及**[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py)**等技术。**[llm.c project](https://github.com/karpathy/llm.c/pull/170)**进行了优化,减少了内存使用并提升了注意力机制的速度。

*   AI生态系统随新平台和资金注入而持续扩大: AI初创公司领域出现诸多活跃事件,包括**[theaiplugs.com](http://theaiplugs.com/)**作为AI插件和助手的市场平台上线,以及**[SpeedLegal在Product Hunt推出](https://www.producthunt.com/posts/speedlegal)**。一份涵盖550轮融资、总额达**[$30亿的AI初创公司融资数据集](https://www.frontieroptic.com/ai-hype-train)**也被整理共享。**[Cohere](https://txt.cohere.com/compressed-embeddings-command-r-plus/)**和**[Replicate](https://replicate.com/docs/billing)**等平台推出新模型和定价结构,显示该生态系统正日趋成熟。

* * *

