Lilian Weng's Video Diffusion Blog
=========================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-lilian-weng-on-video-diffusion/](https://buttondown.email/ainews/archive/ainews-lilian-weng-on-video-diffusion/) 

我们在周末的匆忙中遗漏了一件重要事项,那就是Lilian Weng的[Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)。尽管她的工作很少引发热议,但它几乎总是就给定重要AI主题提供最有价值的资讯,即使她恰好就职于OpenAI。

对于对Sora充满热情的人来说,这是今年迄今最大的AI发布(据传[正在进入Adobe Premiere Pro](https://twitter.com/legit_rumors/status/1779951008539345140)),都应该仔细阅读这篇文章。

![image.png](https://assets.buttondown.email/images/bfc4ad22-23f2-4c2d-8fe5-a8abb87411f3.png?w=960&fit=max)

![image.png](https://assets.buttondown.email/images/1ee16364-c6ae-4168-a7ab-955de5218bc9.png?w=960&fit=max)

按照Lilian的风格，她带领我们深入探索过去2年所有最先进(SOTA)的视频生成技术:

![image.png](https://assets.buttondown.email/images/741bfc10-624e-4a05-b3ab-cb1b45d083d7.png?w=960&fit=max)

意外发现来自她对"Training-free Adaptation"的强调,这正如其听起来一样令人兴奋:

> 令人惊讶的是，可以在不进行任何训练的情况下，将预训练的文本到图像模型适配到视频生成，而不需要任何额外训练🤯。

![image.png](https://assets.buttondown.email/images/ca8e24fc-ffe2-4a93-b6e1-cac34b0ef23f.png?w=960&fit=max)

很遗憾她仅用了2个句子来讨论SORA,但她显然知道更多无法透露的内容。总的来说,这可能是关于SOTA AI视频实际工作原理最权威的解释,除非Bill Peebles再次投笔从戎。

* * *


AI Reddit Recap
===============


**AI Companies and Releases**

*   **OpenAI拓展业务范围**：[OpenAI在日本建立工作室](https://openai.com/blog/introducing-openai-japan)，推出[Batch API](https://i.redd.it/22uslfxivouc1.png)，并与Adobe合作将[Sora视频模型引入Premiere Pro](https://www.youtube.com/watch)。
* 新发布的[Reka Core多模态语言模型](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model)。
*   **竞争格局**：Sam Altman表示OpenAI将["碾压"初创公司](https://twitter.com/ai_for_success/status/1779930498623742187)。Devin AI模型则呈现[创纪录的内部使用量](https://twitter.com/SilasAlberti/status/1778623317651706237)。

**New Model Releases and Advancements in AI Capabilities**

*   **WizardLM-2已发布**: 在 /r/LocalLLaMA 中,WizardLM-2刚刚推出,并展现出[**出色的性能表现**](https://www.reddit.com/r/LocalLLaMA/comments/1c4qi12/wizardlm2_just_released_impressive_performance/)。
*   **Llama 3即将有新消息公布**：一篇[图像帖子](https://i.redd.it/dgt5sbgqfouc1.jpeg)暗示Llama 3的相关新消息即将推出。
* **Reka Core多模态模型发布**：[Reka AI宣布发布了其新型前沿级多模态语言模型Reka Core](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model)。
* **AI模型展现直觉和创造力**：Geoffrey Hinton 表示，当前的 AI 模型正在展现[**直觉、创造力，并能看到人类无法察觉的类比**](https://x.com/tsarnick/status/1778524418593218837)。
* **AI 为自身发展做出贡献**: [Devin 首次成为其自身代码仓库的最大贡献者](https://twitter.com/SilasAlberti/status/1778623317651706237)，一个 AI 系统显著地对其自身代码库做出了贡献。
* 在/r/singularity中分享了[**Opus能识别其自身生成的输出**](https://www.reddit.com/r/singularity/comments/1c4tfnc/opus_can_recognize_its_own_outputs/)，这是一项令人瞩目的新功能。

**Industry Trends, Predictions and Ethical Concerns**

* 关于AI破坏性的警告：Sam Altman [警告初创公司,如果不能够及时适应,可能会被OpenAI碾压](https://twitter.com/ai_for_success/status/1779930498623742187)。
*   **通用人工智能 (AGI) 时间线争议**：尽管 Yann LeCun 认为 AGI 是不可避免的，但他 [表示这不会在明年就实现，也不会仅来自于大型语言模型 (LLMs)](https://x.com/ylecun/status/1779845304788955292)。
*   **模型存在毒性问题**: [WizardLM-2在发布后不久不得不被删除](https://i.redd.it/lyaop5lw0suc1.png)，这是因为开发人员忘记对其进行毒性测试,这突出了负责任的AI开发所面临的种种挑战。
* 美国拟定的AI监管政策：美国AI政策中心提出了[一项新的法案提议来规范美国的AI开发](https://twitter.com/neil_chilson/status/1777695468656505153)。
* 对于AI初创企业,/r/Singularity上的一份[PSA文章](https://www.reddit.com/r/singularity/comments/1c566i0/psa_beware_of_startups_that_looks_too_good_to_be)警示人们要谨慎对待那些看起来过于美好的企业,因为其中一些企业有不明确的加密货币历史。

**Technical Discussions and Humor**

* 构建"专家组合"模型：/r/LocalLLaMA分享了一个关于如何[使用mergoo轻松构建自己的MoE语言模型](https://www.reddit.com/r/LocalLLaMA/comments/1c4gxrk/easily_build_your_own_moe_llm/)的指南。
*   **扩散模型与自回归模型**：/r/MachineLearning 有一个讨论, 比较了[扩散和自回归图像生成方法](https://www.reddit.com/r/MachineLearning/comments/1c53pc5/diffusion_versus_autoregressive_models_for_image/), 并讨论哪种方法更佳。
* 微调 GPT-3.5：/r/OpenAI 发布了一份[微调 GPT-3.5 以满足定制使用场景的指南](https://www.reddit.com/r/OpenAI/comments/1c4j6n7/finetuning_gpt35_for_custom_use_cases/)。

* * *

AI Twitter Recap
================


**WizardLM-2 Release and Withdrawal**

*   **WizardLM-2 发布**: [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899325868589372) 宣布发布下一代先进 LLM 系列产品 WizardLM-2,包括 **8x22B、70B 和 7B 等模型**,与业界领先的专有 LLM 相比,性能表现十分出色。
*   **毒性测试遗漏**：[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1780101465950105775) 对在发布过程中意外遗漏必要的毒性测试而致歉,并表示将尽快完成测试,并尽快重新发布该模型。
* [@abacaj](https://twitter.com/abacaj/status/1780090189563486691)表示，**WizardLM-2模型权重已从Hugging Face拉取**，并猜测这可能是一次过早的发布或其他一些事情正在发生。

**Reka Core Release**

* **Reka Core公告**：[@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894622334189592)宣布发布了他们目前最强大的多模态语言模型 - Reka Core,该模型具有**包括理解视频在内的诸多功能**。
* **技术报告**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894626083864873) 发布了一份详细介绍 Reka 模型的训练、架构、数据和评估的技术报告。
*   **基准性能**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894623848304777) 针对文本和多模态标准基准评估了 Core, 并进行了第三方盲人人工评估, 结果表明其**接近先锋级模型如 Claude3 Opus 和 GPT4-V 等**。

**Open Source Model Developments**

*   **Pile-T5**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1779891910871490856)宣布发布 Pile-T5，这是一个基于T5架构的模型，**采用Llama分词器从Pile数据集训练得到的2T tokens**，具有中间检查点，在基准测试性能上有显著提升。
* **Idefics2**: [@huggingface](https://twitter.com/huggingface/status/1779922877589889400)发布了Idefics2，这是一款**具有显著增强功能的8B视觉-语言模型**，在OCR、文档理解和视觉推理等方面表现出色，并采用Apache 2.0许可证。
* [@SnowflakeDB](https://twitter.com/SnowflakeDB/status/1780225794402627946) 开源了 snowflake-arctic-embed 系列强大的嵌入模型，**参数范围从 22 到 335 百万，嵌入维度为 384-1024，MTEB 评分为 50-56**。

**LLM Architecture Developments**

*   **Megalodon架构**：[@\_akhaliq](https://twitter.com/_akhaliq/status/1780083267888107546)分享了Meta发布的Megalodon,这是一种**高效的大型语言模型(LLM)预训练和推理架构,具有无限的上下文长度**。
*   **TransformerFAM**: [@_akhaliq](https://twitter.com/_akhaliq/status/1780081593643647022)分享了谷歌宣布推出TransformerFAM的消息,其中**Feedback Attention被用作工作内存,使Transformer能够无限长地处理输入**。

**Miscellaneous Discussions**

*   **人形机器人预测**: [@DrJimFan](https://twitter.com/DrJimFan/status/1780254247650787512) 预测, 在未来10年内, **人形机器人的数量将超过 iPhones**, 先是逐步增长, 然后突然激增。
*   **验证码和机器人**: [@fchollet](https://twitter.com/fchollet/status/1780042591440134616)认为**验证码无法阻止机器人注册服务**, 因为专业的垃圾邮件操作会雇佣人工以每个账户约 1 美分的价格手动解决验证码。

* * *

AI Discord Recap
================


**1\. New Language Model Releases and Benchmarks**

*   **[EleutherAI](https://blog.eleuther.ai/pile-t5/)** 发布了 **[Pile-T5](https://github.com/EleutherAI/improved-t5)**，这是一个经过增强的 T5 模型，它是在 Pile 数据集上训练的，最多包含 2 万亿个 token，在基准测试中展现出了更优异的性能。此次发布也在[Twitter 上宣布](https://x.com/arankomatsuzaki/status/1779891910871490856)了。

*   **[Microsoft](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)** 发布了状态先进的指令跟随模型 **[WizardLM-2](https://wizardlm.github.io/WizardLM2/)**,后因未通过毒性测试而 [被移除](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png),但在 [Hugging Face](https://huggingface.co/alpindale/WizardLM-2-8x22B) 等平台仍有镜像保留。

* **[Reka AI](https://publications.reka.ai/reka-core-tech-report.pdf)** 推出了前沿级多模态语言模型**[Reka Core](https://www.youtube.com/watch)**，与 OpenAI、Anthropic 以及 Google 的模型相媲美。

*   **[Hugging Face](https://huggingface.co/blog/idefics2)** 发布了一个名为 **[Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b)** 的8B多模态模型,该模型在视觉语言任务（如光学字符识别、文档理解和视觉推理）方面表现出色。

* 关于模型性能、如**[MinP/DynaTemp/Quadratic](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/)**等采样技术,以及根据**[Berkeley paper](https://arxiv.org/abs/2404.08335)**的分词影响进行了探讨。

**2\. Open Source AI Tools and Community Contributions**

* **[LangChain](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction)** 引入了[重新设计的文档结构](https://discord.com/channels/1038097195422978059/1058033358799655042/1229820483818623010)，并见证了社区贡献,如[Perplexica](https://github.com/ItzCrazyKns/Perplexica/)(一个开源的AI搜索引擎)、[OppyDev](https://oppydev.ai)(一个AI编码助手)和[Payman AI](https://www.paymanai.com/)(可使AI代理雇佣人类)等。

*   **[LlamaIndex](https://twitter.com/llama_index/status/1779898403239125198)** 发布了关于智能代理接口的教程、一个[结合Qdrant引擎的混合云服务](https://twitter.com/llama_index/status/1780275878230139293)，以及一个针对混合搜索的[Azure AI集成指南](https://twitter.com/llama_index/status/1780324017083400235)。

* **[Unsloth AI](https://github.com/unslothai/unsloth/wiki)** 观察到有关 LoRA fine-tuning、ORPO optimization、CUDA 学习资源及清洁 **ShareGPT90k** 数据集用于训练的讨论。

*   **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files)** 提供了多节点分布式微调的指南，而 **[Modular](https://github.com/venvis/mojo2py)** 则介绍了将 Mojo 代码转换为 Python 的 mojo2py 工具。

*   **[CUDA MODE](https://github.com/cuda-mode/lectures/tree/main/lecture%2014)** 提供共享讲座录像，重点介绍CUDA优化技术、**[HQQ+](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py)** 量化技术以及用于高效内核的llm.C项目。

**3\. AI Hardware and Deployment Advancements**

* 对于[Nvidia潜在的早期RTX 5090发布](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch)的讨论,则源于市场竞争压力以及预期的性能提升。

*   **[Strong Compute](https://strongcompute.com/research-grants)** 宣布为探索 AI 的可信度、后 Transformer 架构、新的训练方法和可解释 AI 的 AI 研究人员提供 **$10k - $100k** 的资助，并提供 GPU 资源。

* **[Limitless AI](https://x.com/dsiroker/status/1779857843895599383)**, 曾被称为 Rewind, 推出了一款可穿戴 AI 设备, 引发了围绕数据隐私、HIPAA 合规性和云存储等问题的讨论。

*   **[tinygrad](https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455)** 探索了成本效益的 GPU 集群设置、MNIST 数据处理、文档改进,并在过渡到 1.0 版本的过程中增强了开发人员体验。

* [**将自定义模型打包到llamafiles中(https://github.com/Mozilla-Ocho/llamafile/pull/59)**]、在消费者硬件上运行CUDA以及使用tinygrad将ONNX模型转换至WebGL/WebGPU。

**4\. AI Safety, Ethics, and Societal Impact Debates**

* 对 AI 开发的伦理影响进行讨论，包括需要制定 [**安全基准**] 如 [ALERT](https://github.com/Babelscape/ALERT) 来评估语言模型可能产生的有害内容。

* 对于虚假信息传播和不道德做法的担忧,其中提及在Facebook上广告的一个可能的AI欺诈方案,名为[Open Sora](https://www.open-sora.org)。

* 关于在AI能力和社会期望之间寻求平衡的辩论,一些人主张创造性自由,而另一些人则优先考虑安全因素。

* 哲学性交流比较了 AI 系统与人类的推理能力, 涉及独立决策、情商和语言理解的神经生物学基础等方面。

* 针对DeepFake和生成AI内容的新兴法律法规 [**与**] 执法挑战和意图考量相关的讨论正在展开。

**5\. Misc**

*   **新模型引发的兴奋与猜测**：[EleutherAI](https://blog.eleuther.ai/pile-t5/)推出的**Pile-T5**、[Hugging Face](https://huggingface.co/HuggingFaceM4/idefics2-8b)的**Idefics2 8B**、[Reka AI](https://publications.reka.ai/reka-core-tech-report.pdf)的**Reka Core**以及微软的**WizardLM 2**(尽管它遭到神秘的[删除](https://fxtwitter.com/pimdewitte/status/1780066049263538653))等新 AI 模型的发布引起了广泛关注和探讨。AI 社区积极探索这些模型的功能与训练方法。

* 多模态AI和扩散模型的进步：对话突出了像[IDEFICS-2](https://huggingface.co/blog/idefics2)这样的**Multimodal AI**模型在OCR、视觉推理和对话能力方面的进步。对**视频生成的Diffusion模型**([Lilian Weng的博客文章](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/))以及**语言建模中的分词重要性**([UC Berkeley论文](https://arxiv.org/abs/2404.08335))的研究也引起了兴趣。

* 模型开发的工具和框架：讨论涵盖了用于AI开发的各种工具和框架,包括**Axolotl**用于[multi-node分布式微调](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files)、**LangChain**用于[构建LLM应用程序](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction)、**tinygrad**用于[高效深度学习](https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455),以及**Hugging Face的库**如[parler-tts](https://github.com/huggingface/parler-tts)用于高质量TTS模型等。

*   **新兴平台和倡议**：AI界注意到了各种新兴平台和倡议,如个性化AI的**Limitless**（[从Rewind重新命名](https://x.com/dsiroker/status/1779857843895599383)）、用于[多方面数据搜索](https://txt.cohere.com/compass-beta/)的**Cohere Compass**测试版、用于[AI到人工任务市场](https://www.paymanai.com/)的**Payman AI**,以及**Strong Compute**提供[10k-100k美元的AI研究资助](https://strongcompute.com/research-grants)。这些发展彰显了应用AI生态系统日益扩大。

* * *

