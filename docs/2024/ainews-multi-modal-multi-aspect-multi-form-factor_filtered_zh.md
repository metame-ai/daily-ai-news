Multi-Model: Reka Core, IDEFICS
==========================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-multi-modal-multi-aspect-multi-form-factor/](https://buttondown.email/ainews/archive/ainews-multi-modal-multi-aspect-multi-form-factor/) 

在AI领域,一些天里可以发生整整一个月的事情 - 就像[2月15日出现了Sora和Gemini 1.5](https://buttondown.email/ainews/archive/ainews-sora-pushes-sota/)以及其他一系列发布一样,4月见证了以下机构的重大发布:

[Reka Core](https://twitter.com/RekaAILabs/status/1779894622334189592?utm_source=ainews&utm_medium=email&utm_campaign=ainews-multi-modal-multi-aspect-multi-form-factor)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------

一个新型的 GPT4 类多模态基础模型...

![image.png](https://assets.buttondown.email/images/908de291-3f84-4b64-b9eb-75a5deb689a5.png?w=960&fit=max)

具有实际参考价值的技术报告

![image.png](https://assets.buttondown.email/images/91cc1830-d65f-403c-8c71-40e1d0262894.png?w=960&fit=max)

被称为"full Shazeer"

![image.png](https://assets.buttondown.email/images/15eadb75-acea-48bf-8bb1-7644279acb3c.png?w=960&fit=max)

Cohere Compass
--------------

> 我们新推出的基础嵌入模型可对multi-aspect数据进行索引和搜索。多方面数据可理解为包含多个概念和关系的数据。这在企业数据中很常见 - 电子邮件、发票、简历、支持工单、日志消息和表格数据都包含大量具有上下文关联的内容。

![image.png](https://assets.buttondown.email/images/64c761c1-7926-4dcd-bace-e5844f7c5c3d.png?w=960&fit=max)

IDEFICS 2-8B
------------

继续从[去年的IDEFICS](https://www.latent.space/p/idefics)开展工作,这是一个完全开源的谷歌未发布的Flamingo多模态模型的复制版本。

![image.png](https://assets.buttondown.email/images/05186c69-c132-4ecc-ac5a-8cd4fe44ac77.png?w=960&fit=max)

重回Limitless应用场景
--------------------------

> 这是一款网页应用程序、Mac应用程序、Windows应用程序以及可穿戴设备。

[间谍软件已退出时代, Pendants 正处于流行期]。(https://twitter.com/dsiroker/status/1779857843895599383)

![image.png](https://assets.buttondown.email/images/799f8934-9579-4e24-b114-9af2ed4ca3d8.png?w=960&fit=max)

* * *


AI Reddit Recap
===============


**AI Models and Performance**

*   **Apple MLX性能**：在 /r/LocalLLaMA 中，Apple MLX (0.10.0) 在 M2 Ultra 上达到了 [**105.5 tokens/s，击败了使用 Mistral Instruct 4bit 的 Ollama/Llama.cpp 的 95.1 tokens/s**](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/)。
*   **Ollama性能对比**: 在/r/LocalLLaMA中,使用Mistral Instruct 0.2 q4_0的Ollama性能显示[**M2 Ultra 76GPU以95.1 t/s处于领先**](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/),其次是Windows Nvidia 3090的89.6 t/s,WSL2 NVidia 3090的86.1 t/s,以及M3 Max 40GPU的67.5 t/s。Apple MLX在M2 Ultra上达到103.2 t/s,在M3 Max上达到76.2 t/s。
* 在 /r/LocalLLaMA 上, [**M3 Max 64GB 在使用 Command-R 和 Dolphin-Mixtral 8x7B 模型处理长上下文时, 其提示速度比 M1 Max 64GB 快超过一倍**](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/)。
* 针对 LLMs 的 GPU 注意事项：在 /r/LocalLLaMA 上，一位用户正在寻求建议，是选择围绕 [RTX 4090 构建机器](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/)还是购买 MAC 来运行像 Command R 和 Mixtral 这样的 LLMs 模型，并具有未来的可升级性。
* 在/r/StableDiffusion上的一篇文章中比较了[3060 12GB与4060 16GB在Stable Diffusion中的表现](https://www.reddit.com/r/StableDiffusion/comments/1c3nk9g/3060_12gb_vs_4060_16gb_for_sdxl_or_wait_for_50xx/)，建议选择尽可能多的显存容量。4060ti 16GB在进行SDXL 20步骤时需要18.8秒。

**LLM and AI Developments**

*   **将AI与人类进行比较**：在 /r/singularity 上，微软研究院的Chris Bishop将吐出信息的AI模型比喻为"随机鹦鹉"，指出[**人类做同样事情时会被授予大学学位**](https://www.reddit.com/r/singularity/comments/1c3o1o2/microsoft_researchs_chris_bishop_when_ai_models/)。评论探讨了学位的有效性以及它们是否仅仅表示信息的重复吐出。
*   **对工作的影响**：前PayPal CEO Dan Schulman预测，[**GPT-5将成为一个"惊慌失措"的时刻，并且80%的工作将缩减80%范围**](https://twitter.com/woloski/status/1778783006389416050)。
*   **对AGI的执著**: Mistral的CEO Arthur Mensch认为[对实现AGI的执著是关于"创造神"](https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4)。Gary Marcus在推特上也[敦促不要创造有意识的AI](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/)。
* 为未来而建设：Sam Altman在推特上发文提到[人们正在努力开发供未来世代持续推进的技术](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/)。
* 一篇发在 /r/singularity 上的文章认为[实现 AGI 将迫使人类正视其对于什么创造意识的理解存在缺乏](https://www.reddit.com/r/singularity/comments/1c3yvs1/agi_will_cause_humans_to_confront_that_they_do/)，预测围绕 AI 伦理和权利会引发争论和紧张局势。

**Industry and Career**

* 在 /r/MachineLearning 上，一名博士研究生在被聘为一个与实际 ML 工作无关的职位后，[询问如何发现"假"的 ML 角色](https://www.reddit.com/r/MachineLearning/comments/1c3z8ug/d_advice_for_spotting_fake_ml_roles/)。他指出，在面试中提出问题可能无法有效分辨,因为存在潜在的不诚实。
* 另一名博士生[质疑了在LLM时代, 文本分类、命名实体识别和关系提取等传统NLP任务的重要性](https://www.reddit.com/r/MachineLearning/comments/1c4a7sa/d_are_traditional_nlp_tasks_such_as_text/)，对自己的研究前景感到担忧。
*   **LLM 的实际应用**：/r/MachineLearning 上的一篇帖子[要求提供 LLM 在文本生成之外的行业实际应用示例](https://www.reddit.com/r/MachineLearning/comments/1c4cr32/d_in_industry_nlp_are_there_any_actualpractical/)，这些应用能带来良好的投资回报率。该帖子指出，像语义搜索等任务可以由其他模型很好地处理。

**Tools and Resources**

* **DBRX在llama.cpp中的支持**：[Llama.cpp现已支持DBRX](https://github.com/ggerganov/llama.cpp/pull/6515)，这是一种用于大语言模型的二进制格式。
*   **更快的结构化生成**: 在 /r/LocalLLaMA 中，[一种新的用于 LLMs 中结构化生成的方法](https://www.reddit.com/r/LocalLLaMA/comments/1c3oa8f/faster_than_llamacpps_grammar_structured/)声称比 llama.cpp 的方法快许多，其运行时间与语法复杂度或模型/词汇表大小无关。作者计划在不久后开源这种方法。
*   **Python数据排序工具**：作者已[开源他们的Python工具集](https://github.com/nazpins/naztech-automated-data-sorting-tools)，旨在自动化数据排序和组织，以高效处理无序文件和大量数据。
* 一个[开源的简单PyTorch离散扩散实现（400行代码）](https://www.reddit.com/r/MachineLearning/comments/1c3pvx5/p_extremely_short_and_simple_implementation_of/)已在/r/MachineLearning上分享。这是一个**简单离散扩散实现**。

**Hardware and Performance**

*   **M1 Max vs M3 Max**：在 /r/LocalLLaMA 中进行的[对比实验](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/)显示，M3 Max 配备 64GB RAM 时的提示速度超过 M1 Max 一倍以上，特别是在处理长文本场景下。
*   **RTX 4090 与苹果电脑对比**: 一篇文章询问[是建立一台搭载RTX 4090的电脑,还是购买苹果电脑运行大语言模型](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/)的建议,认为电脑会更便宜且更易于升级。
* Apple的MLX库[在拥有76个GPU核心的M2 Ultra上达到了105.5 tokens/s](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/)，超过了llama.cpp在运行Mistral Instruct 4-bit模型时的95.1 tokens/s。
*   **Ollama性能对比**：一份[对Ollama库在不同硬件上的性能对比](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/)显示，M2 Ultra以95.1 t/s的成绩居首位，其次为Windows Nvidia 3090、WSL2 Nvidia 3090和使用Mistral Instruct模型的M3 Max。

* * *

AI Twitter Recap
================


**AI Models and Architectures**

*   **新模型发布**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894622334189592)宣布推出Reka Core，这是他们"迄今最佳且最强大的多模态语言模型"，与GPT-4/Opus级模型相媲美。[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899325868589372)发布了WizardLM-2系列模型,包括8x22B、70B和7B等多个变体,与业内领先的大型语言模型(LLM)相竞争。
*   **架构和训练**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1779684686618628323)指出**Transformer是一种常用于扩散算法的架构**。[@ylecun](https://twitter.com/ylecun/status/1779845304788955292)表示AI最终将超越人类智能,但这不会仅靠当前的自回归型大语言模型(Auto-regressive LLMs)实现。
*   **优化与扩展**: [@karpathy](https://twitter.com/karpathy/status/1779272336186978707)优化了一个LLM模型(Large Language Model),使其在C中与PyTorch的性能相匹配,现在每次迭代仅需26.2ms,采用了诸如在fp32模式下使用cuBLAS等技巧。[@_lewtun](https://twitter.com/_lewtun/status/1779804085677404583)认为强大的Mixtral-8x22B模型微调将能弥补与专有模型之间的差距。

**AI Capabilities and Benchmarks**

*   **多模态能力**: [@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894626083864873) 分享了 Reka Core 的视频理解能力,在多模态对话中胜过了 Claude3 Opus。[@DrJimFan](https://twitter.com/DrJimFan/status/1779558822543229221) 推测特斯拉 FSD v13 可能利用语言 token 来推理复杂的自动驾驶场景。
*   **编码与数学**: [@OfirPress](https://twitter.com/OfirPress/status/1779195498328429045)注意到开源编码 agent SWE-agent在10天后已拥有1.5k用户。[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899333678387318)使用完全由AI驱动的合成训练系统来改进 WizardLM-2。
*   **基准测试和排行榜**: [@svpino](https://twitter.com/svpino/status/1779185295541575710)注意到在 GPT-4 更新之前, Claude 3 曾于人工评估排行榜上短暂占据最佳模型地位。 [@bindureddy](https://twitter.com/bindureddy/status/1779186163464708314)分析了 GPT-4 在编码、数学和知识方面的性能表现。

**Open Source and Democratizing AI**

*   **开放模型和数据**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1779891910871490856) 宣布了采用来自Pile的2T个tokens和Llama分词器的EleutherAI Pile-T5模型。[@_philschmid](https://twitter.com/_philschmid/status/1779922877589889400) 介绍了一款小于10B参数的开源视觉语言模型(VLM) Idefics2,具有出色的OCR、文档理解和视觉推理能力。
*   **可访问性和成本**：[@maximelabonne](https://twitter.com/maximelabonne/status/1779801605702836454)指出,开源模型与顶级封闭源模型的差距现已缩短至6-10个月,而非数年。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1779805019841142791)预测,到年底这一差距将完全消除,因为开源方案在大多数应用场景下更快、更便宜且更安全。
*   **计算与工具**: [@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899329844760771) 正在在Hugging Face上分享8x22B和7B的 WizardLM-2权重。[@aidangomez](https://twitter.com/aidangomez/status/1779882113573044625) 宣布了用于多方面数据搜索的 Compass 嵌入模型测试版。

**Industry and Ecosystem**

* 公司扩张: [@gdb](https://twitter.com/gdb/status/1779762694473551924)和[@hardmaru](https://twitter.com/hardmaru/status/1779783633961935218)指出,OpenAI进驻日本是AI领域的重要发展。[@adcock_brett](https://twitter.com/adcock_brett/status/1779541107577151916)分享了加拿大240亿加元投资于AI能力和基础设施的消息。
*   **新兴应用**：[@svpino](https://twitter.com/svpino/status/1779843933276672195) 利用Langflow的可视化界面和Langchain,无需编码即可构建了一个完整的RAG应用。[@llama_index](https://twitter.com/llama_index/status/1779542320133947622) 展示了如何运用LLMs和知识图谱加速生物材料的发现。
*   **道德考量**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1779171248083177500) 在 Facebook 期间未能对 @PalmerLuckey 给予更多支持而感到遗憾。[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1779721105407873411) 赞扬该视频是迄今为止最佳的 AI 存在风险报道。

* * *

AI Discord Recap
================


大型语言模型(LLM)的进展：各个平台和组织围绕LLM的新版本及功能变革,表现出了强烈的兴趣和讨论。主要示例包括:

* **[Pile-T5](https://blog.eleuther.ai/pile-t5/)** 是 EleutherAI 开发的一个 T5 模型变体,该模型在 2 万亿个 token 数据上进行了训练,在 SuperGLUE 和 MMLU 等基准测试中表现有所提升。所有相关资源,包括模型权重和脚本,均[已在 GitHub 上开源](https://github.com/EleutherAI/improved-t5)。

*   **[WizardLM-2](https://wizardlm.github.io/WizardLM2/)** 系列正式发布,包括 8x22B、70B 和 7B 等不同规模的模型,这引发了人们在 OpenRouter 上部署的热情期待。与 GPT-4 相比,WizardLM-2 8x22B 表现出色。

* **[Reka Core](https://publications.reka.ai/reka-core-tech-report.pdf)** 是来自 Reka AI 的前沿级**多模态语言模型**，相关训练、架构和评估的详细信息已在技术报告中分享。
2. **LLM训练和推理的优化及技术**：广泛讨论围绕优化LLM开发的各个环节进行，包括:

* 通过采用 [Ring Attention](https://coconut-mode.com/posts/ring-attention/) 等方法实现高效的上下文处理，从而使模型能够借助多台设备扩展至几乎无限的上下文窗口。

*   利用**Model compression**技术（如**LoRA**、**QLoRA**和**16-bit quantization**）来减小内存占用，这些见解来自于[Lightning AI](https://lightning.ai/pages/community/lora-insights/)和社区实验。

*   采用**硬件加速**策略,如利用[tinygrad的驱动补丁](https://github.com/tinygrad/open-gpu-kernel-modules)在NVIDIA 4090 GPUs上开启**P2P支持**,取得了显著的性能提升。

* 在框架如[LLM.c](https://github.com/karpathy/llm.c)和[torchao](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)中探索高效的张量布局、填充和矩阵运算的 swizzling, 对 **Kernel 优化** 进行研究。
3. **开源倡议和社区协作**：AI社区展现了对开源开发和知识共享的坚定承诺,体现在:

*   **开源** 重大项目如 [Pile-T5](https://github.com/EleutherAI/improved-t5)、[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 和 [Mixtral](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/io.mojo) (在 Mojo 中)，促进透明度和协作努力。

* 教育资源,如[CUDA MODE讲座](https://github.com/cuda-mode/lectures)以及招募志愿者录制和分享内容的倡议,有助于促进知识传播。

*   **社区项目**，如 [llm.mojo](https://github.com/dorjeduck/llm.mojo)（llm.c 的 Mojo 移植版）、[Perplexica](https://github.com/ItzCrazyKns/Perplexica/)（Perplexity AI 复制品）和 [LlamaIndex integrations](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122) 用于文档检索, 展现了草根创新。
4. **Datasets and Data Strategies for LLM Development**：讨论强调了数据质量、数据整理以及针对训练数据的战略方法的重要性，包括：

* 采用诸如[StableLM](https://arxiv.org/abs/2402.17834)(ReStruct数据集)和[MiniCPM](https://arxiv.org/abs/2404.06395)(结合OpenOrca和EvolInstruct)等技术进行**合成数据生成**,重点关注使用定制合成数据来对齐大型语言模型的[CodecLM](https://arxiv.org/pdf/2404.05875.pdf)技术。

*   **数据filtering策略**及[数据curation scaling laws](https://x.com/pratyushmaini/status/1778577153107570770)的开发,强调curation不能与计算无关,正如CVPR 2024论文中所述。

* 多语言与多模态数据集，呼吁获得版权许可的EU文本及多模态数据，以训练大型开放多模态模型，这反映了对多样化数据源的不断增长需求。[Source 1](https://blog.eleuther.ai/pile-t5/) | [Source 2](https://wizardlm.github.io/WizardLM2/) | [Source 3](https://publications.reka.ai/reka-core-tech-report.pdf) | [Source 4](https://coconut-mode.com/posts/ring-attention/)
5. **其他**

*   **Stable Diffusion 3引发兴奋和争论**: AI界正期待着**[Stable Diffusion 3 (SD3)](https://www.youtube.com/watch)**的发布,讨论其在质量和效率方面的潜在改进。讨论集中在如何利用**[SD Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)**等工具优化在性能较低GPU上的性能表现,以及探索使用**[ControlNet](https://github.com/lllyasviel/ControlNet)**、**Lora**和**outpainting**等AI驱动创作工艺的新颖应用场景。SD3中严格的提示词审查引发了关于潜在质量下降的担忧,这在[Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/)上有所讨论。

*   **Perplexity AI的路线图和模型对比**：**Perplexity AI** 6月份的路线图预告了新功能,包括强制实施JSON语法、新的Databricks模型、模型信息端点、状态页面和多语言支持,可在其[功能路线图页面](https://docs.perplexity.ai/docs/feature-roadmap)查看。讨论比较了模型如**Claude Opus**、**GPT-4**和**RAG**在不同任务中的上下文窗口和性能表现。Meta在WhatsApp上推出了一个类似Perplexity AI的AI界面,引发人们对AI在即时通讯平台不断集成的关注,正如[该文章](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/)所报道的。

*   **Tinygrad在NVIDIA GPU上实现P2P**: **Tinygrad**通过修改NVIDIA驱动程序,成功在**NVIDIA 4090和4070 TI Super GPU**上启用了点对点(P2P)支持,实现了 14.7 GB/s 的AllReduce性能。这一突破已在[Twitter](https://twitter.com/__tinygrad__/status/1778677126092509611)上分享,代码可在[GitHub](https://github.com/tinygrad/open-gpu-kernel-modules)获取,对于降低运行大型语言模型的成本具有重要意义。**CUDA社区**正在通过[One Billion Row Challenge](https://1brc.dev/)推动性能边界,并探索低精度运算的优化策略。
* 基于Eleuther AI的Pile-T5模型和研究洞见：EleutherAI推出了[Pile-T5](https://blog.eleuther.ai/pile-t5/)，这是一个基于T5模型的变体，它在Pile的2万亿个词tokens上进行了训练，在SuperGLUE和MMLU等基准测试中展现出了优秀的性能。这个模型在与代码相关的任务上表现出色，其权重和训练脚本已在[GitHub上开源](https://github.com/EleutherAI/improved-t5)。研究讨论深入探讨了MoE与密集型Transformer模型的性能对比、语言模型中分词的作用，以及利用Google的[Patchscopes](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/)框架解释隐藏表示的可解释性。

* * *

