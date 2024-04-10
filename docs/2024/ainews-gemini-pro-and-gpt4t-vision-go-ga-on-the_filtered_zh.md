Gemini Pro和GPT4T Vision在同一天完全偶然上线API。
====================================================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the/](https://buttondown.email/ainews/archive/ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the/) 


*   在[Google Cloud Next](https://cloud.withgoogle.com/next?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)的第一天, **拥有百万个token上下文窗口的[Gemini 1.5 Pro 从等待名单中释放出来](https://x.com/OfficialLoganK/status/1777733743303696554?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)** 并在180多个国家自由使用。此外, 它还可以:
    *   理解最多 9.5 小时的音频 ([quote](https://twitter.com/liambolling/status/1777758743637483562?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the): "不仅仅是你说的话,还有音频背后的语调和情感。在某些情况下,它甚至能理解诸如狗叫和下雨之类的声音。")
    * 使用新的[File API](https://ai.google.dev/tutorials/prompting_with_media?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)上传几乎无限的文件,并且免费。
    * Gecko-1b-256/768模型，又称text-embedding-004模型,是一种新的小型embedding模型,在MTEB上的性能优于同类规模的模型。
    * JSON模式和更好的函数调用
*   3小时后, [GPT-4 Turbo with Vision现已在API中普遍可用](https://twitter.com/OpenAIDevs/status/1777769463258988634?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)，但其中隐藏着对GPT-4 Turbo语言模型本身的重大更新。
    *   没有博客文章 - 我们知道的只是它已经[majorly improved](https://twitter.com/OpenAI/status/1777772582680301665?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)，而且[reasoning has been further improved](https://x.com/polynoamial/status/1777809000345505801?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)。也许它只是变得非常非常非常擅长[delving](https://x.com/ChatGPTapp/status/1777221658807521695?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)？


* * *


AI Reddit Recap
===============


**Latest AI Model Developments**

*   **Meta Platforms将于下周推出Llama 3的小型版本**：根据[TheInformation](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)报道，Meta计划发布其Llama 3模型的较小版本。(433 upvotes)
*   **Orca 2.5 7B使用新的DNO方法超越了旧版本的GPT-4在AlpacaEval中**: [Orca 2.5](https://huggingface.co/papers/2404.03715?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)通过使用Direct Nash Optimization(DNO)将对比学习与优化一般偏好相结合,从而超越了拥有更多参数的模型。(60 upvotes)
*   **Functionary-V2.4 发布为OpenAI function calling models的替代品**：与OpenAI models相比，[Functionary-V2.4](https://www.reddit.com/r/LocalLLaMA/comments/1bzhyku/nanollava_1b_pocket_size_vlm/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提供了更好的性能和新的代码解释器功能。(20 upvotes)
*   **CosXL - Cos Stable Diffusion XL 1.0和1.0 Edit模型发布**: 这些模型使用Cosine-Continuous EDM VPred调度来实现全色域和指令图像编辑。(9 upvotes)

**Efficient AI Techniques**

*   **[R] 高效扩散模型中的Missing U**: 这篇[研究](https://www.reddit.com/r/MachineLearning/comments/1bzfns4/r_the_missing_u_for_efficient_diffusion_models/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提出使用神经常微分方程替换离散的 U-Net, 在保持质量的同时实现了**最多 80% 的推理加速、75% 的参数减少和 70% 的 FLOP 减少**。(38 upvotes)
*   **[R] 没有指数级数据就没有"零样本"**: 一项[研究发现](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)多模态模型需要指数级更多的预训练数据才能在"零样本"性能上实现线性提升。(12 upvotes)
*   **[R] 高性能语言技术的新型大规模多语种数据集**: [HPLT resources](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)涵盖75种语言,共**约5.6万亿个词汇**,以及18对以英语为中心的并行语言对。

**Creative Applications**

*   **新手教程: 使用Stable Diffusion生成一致的角色面部**: 一个[教程](https://www.reddit.com/gallery/1bzix80?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)使用Automatic1111来生成一致的角色视觉效果。(597个赞)
*   **使用Gemini pro 1.5制作的触摸屏波射游戏, 无需任何代码**: 该游戏是在大约[5个小时](https://v.redd.it/rujagnqzn7tc1?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)内通过告诉Gemini所需的功能而创建的。(189个赞)
* 日本科幻小说作家使用AI创作小说预告片：这个[使用AI生成的预告片](https://v.redd.it/wgrbu0aindtc1?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)展示了一个新颖的应用案例。(20个赞)
*   **引入Steerable Motion 1.3以使用批量图像驱动视频**: 该[新版本](https://www.reddit.com/r/StableDiffusion/comments/1bzakf3/introducing_steerable_motion_13_drive_videos_with/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提供更高的细节、更平滑的运动和更好的控制。(28个赞)

**Scaling AI Infrastructure**

*   **AI公司正在耗尽互联网数据资源**: Models正在[消耗大量](https://lifehacker.com/tech/ai-is-running-out-of-internet?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)的在线数据. (274 upvotes)
*   **[D] 确保加拿大的AI优势**: 加拿大正在投资[2.4十亿美元](https://www.reddit.com/r/MachineLearning/comments/1bytkh8/d_securing_canadas_ai_advantage/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)来加速AI就业增长、提高生产力,并确保负责任的发展,其中包括20亿美元用于AI计算基础设施。
*   **Sam Altman揭示了人工智能的下一步**: Altman在[post](https://www.reddit.com/r/singularity/comments/1bzjcqm/sam_altman_reveals_whats_next_for_ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)中分享了他的愿景。(601个赞)
*   **Sam Altman没有OpenAI的股权，但创业投资使他成为了亿万富翁**: 虽然没有OpenAI的股权，[Altman的投资](https://www.reddit.com/r/singularity/comments/1bzjcqm/sam_altman_reveals_whats_next_for_ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)一直非常赚钱。

**Responsible AI Development**

*   **'AI时代可能出现社会秩序崩溃'，日本两家顶级公司表示**: [WSJ报道](https://www.wsj.com/tech/ai/social-order-could-collapse-in-ai-era-two-top-japan-companies-say-1a71cc1d?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)了日本企业发出的警告。
*   **加拿大AI安全研究所将以5000万美元成立**：加拿大AI投资计划的一部分旨在[进一步促进安全的AI发展](https://www.reddit.com/r/MachineLearning/comments/1bytkh8/d_securing_canadas_ai_advantage/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)。(70个赞)
*   **[D] 对于那些单独发表过论文的人来说,您的经历如何?**: 关于单独发表AI研究论文的可行性和困难性的[讨论](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)。(55个赞)


* * *

AI Twitter 摘要
================


**Cohere Command R+ Model Performance**

*   **Command R+ 攀升至 Arena 排行榜第6名**：[@lmsysorg](https://twitter.com/lmsysorg/status/1777630133798772766?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)指出 Command R+ 已攀升至第6名,通过13K+人类投票达到了与 GPT-4-0314 相同的水平,使其成为**排行榜上最佳的开放模型**。[@seb\_ruder](https://twitter.com/seb_ruder/status/1777671882205962471?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)强调这甚至还没有考虑到 Command R+ 在 RAG、工具使用和多语言能力方面的出色表现。
*   **Command R+在财务RAG中战胜其他模型**：[@virattt](https://twitter.com/virattt/status/1777676354596618474?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)发现，使用OpenAI嵌入、余弦相似度检索、Cohere重排序以及Opus和人工评估，Command R+比Claude Sonnet在财务RAG评估中更快且准确性提高5%。
*   **Command R+是一个拥有先进功能的104B参数模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1777771141886623840?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)指出，Command R+是一个拥有128K个token上下文窗口的104B参数模型,覆盖10种语言,使用工具,并针对RAG进行了专门调优。这是第一个基于Elo评分超过GPT-4的开放权重模型。

**Other Notable Open Model Releases and Updates**

*   **Google发布Code Gemma模型**: [@fchollet](https://twitter.com/fchollet/status/1777715491550994732?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)宣布发布CodeGemma,这是Gemma系列模型的新版本,针对代码生成和完成进行了微调,在2B和7B规模上取得了**最先进的结果**。[@_philschmid](https://twitter.com/_philschmid/status/1777716728921600000?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提供了更多细节,指出这些模型具有8192k的上下文长度,是从Gemma Base初始化的,在额外的500B个Token(网络、代码和数学)上进行了训练,使用SFT和RLHF进行了微调,其中2B模型在HumanEval上达到了27%,7B模型达到了52%。
*   **Google发布了优于transformer的Griffin架构**：[@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1777747790564589844?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)分享说，Google发布了一个采用新的Griffin架构的模型，在MMLU跨不同参数大小的得分和许多基准测试的平均得分上都**优于transformer基线**，同时具有更快的推理和更低内存使用率的效率优势。
*   **Google在Vertex AI上发布Gemini 1.5 Pro**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777738279137222894?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)宣布发布Gemini 1.5 Pro,现已在Google Cloud的Vertex AI平台公开预览,具有**长上下文窗口**,可用于分析大量数据,构建AI客户服务代理等。
*   **DeepMind发布Imagen 2到Vertex AI**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747320945234422?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)宣布他们的生成技术Imagen 2现在可以从一个单一的提示创建**4秒的实时图像**,并可在Google Cloud的Vertex AI平台上使用。
*   **Anthropic推出Constitutional AI模型**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728366101119101?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)发布关于测量模型说服力的新研究,开发了一种测试语言模型说服力的方法,并分析了说服力如何随着Claude的不同版本而变化。
*   **Meta宣布MA-LMM模型**: [@_akhaliq](https://twitter.com/_akhaliq/status/1777539936364662817?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)分享,Meta宣布了用于长期视频理解的MA-LMM (Memory-Augmented Large Multimodal Model),通过大幅减少长上下文长度的GPU内存使用,实现了更长的上下文。

**Emerging Trends and Discussions**

*   **用于代码生成和理解的AI**： 几个讨论围绕着使用AI进行代码生成、理解和调试展开。 [@abacaj](https://twitter.com/abacaj/status/1777574208337215678?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the) 突出了一篇论文,展示了一种方法,每个解决了67个GitHub问题, 而开发人员平均需要花费超过2.77天。 [@karpathy](https://twitter.com/karpathy/status/1777427944971083809?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the) 开源了llm.c,这是一个只有~1,000行代码的纯C语言实现的GPT-2训练。
*   **AI在编码任务中表现优于人类**: 存在多种讨论,探讨AI取代或增强程序员的潜力。[@svpino](https://twitter.com/svpino/status/1777430219785130067?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the) 认为,虽然AI可以将非编码人员转变为平均程序员,并帮助平均程序员变得更好,但目前它可能无法帮助专家级程序员太多,他援引了试图自动化编程的悠久历史、数据和语言本身的局限性,以及技术进步 要到达大众需要一定时间的事实。
*   **语言模型的缩放规律**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1777424149415145882?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the) 分享了关于语言模型缩放规律的详细概述,这些规律允许**从低成本的小规模实验准确预测较大训练集的性能**。该文章涵盖了缩放规律如何适用于过度训练的模型和下游任务性能,以及如何利用它们大幅降低大规模训练运行的计算成本。
*   **DSPy for language model programs**: [@lateinteraction](https://twitter.com/lateinteraction/status/1777731981884915790?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)介绍了DSPy，一种用于构建语言程序的方法论、编程模型和优化器 - **在系统中多次调用LM的任意控制流**。DSPy可以优化LM调用的提示和权重,以最大化给定指标下的程序质量。
*   **语言模型的物理学**: [@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1777638750740210175?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)分享了一篇研究语言模型知识容量扩展规律的论文,估计它们即使量化为int8也可以存储**每参数2比特的知识**,这意味着一个7B模型可以存储140亿比特的知识,超过了英文维基百科和教科书的总和。

* * *

AI Discord 回顾
================


**1\. New AI Model Releases and Capabilities**:

*   Google发布了[Gemini 1.1](https://huggingface.co/chat/models/google/gemma-1.1-7b-it?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)的升级版本,具有编码能力,并引入了专门用于代码任务的[CodeGemma](https://huggingface.co/lmstudio-community?search_models=codegemma&utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)模型。
*   OpenAI推出了[GPT-4 Turbo](https://openai.com/pricing?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the),它具有更大的128k上下文窗口,并且知识更新至2023年12月。
*   Stability AI的[CosXL](https://huggingface.co/stabilityai/cosxl?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)模型需要在非商业研究许可下共享。
*   对于 **Meta的Llama 3** 及其潜在的多模态能力,有越来越大的期待,有关即将发布较小版本的[猜测](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)。

**2\. Efficient LLM Training and Deployment Approaches**:

*   Andrej Karpathy介绍了**[llm.c](https://github.com/karpathy/llm.c?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**，这是一个在1,000行C/CUDA代码中实现的简约版GPT-2训练程序。
*   围绕**low-precision quantization**技术的讨论,如**[HQQ](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the#L808)**用于高效部署大型语言模型(LLM),特别是在移动设备上。
*   Meta 赞助了一项 **[LLM knowledge study](https://arxiv.org/abs/2404.05405?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**，涉及 420 万 GPU 小时的大量计算。
*  Groq提供了**1/10的推理成本**的服务,面向75,000名开发者,可能与Meta的推理能力相抗衡。

**3\. AI Assistants and Multimodal Interactions**:

*   令人兴奋的是**[Gemini 1.5 Pro](https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww&utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**可用于理解音频、通过JSON模式执行命令以及启用多模态AI应用程序。
*  [Syrax AI Telegram bot](https://t.me/SyraxAIBot?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提供聊天历史总结和维护垃圾邮件黑名单等功能。
*   开发者为诸如[虚拟试衣](https://youtu.be/C94pTaKoLbU?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)和创建社交媒体帖子等任务构建了AI Agent。
* 类似[Lepton AI](https://www.lepton.ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)的平台使用Photon和WhisperX等工具简化了AI应用程序的部署。

**4\. Open-Source AI Frameworks and Community Efforts**:

*   LlamaIndex展示了使用[LlamaParse](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提高Retrieval-Augmented Generation (RAG)以及评估高级RAG方法如[ARAGOG](https://twitter.com/llama_index/status/1777441831262818403?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)的技术。
*  Mojo 编程语言 **[开源了其标准库](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**，并提供了[贡献指南](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)供社区参与。
*  Hugging Face引入了**[Gradio的API记录器](https://x.com/abidlabs/status/1775787643324051582?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**,并发布了包含超过2600万页的大规模**[OCR数据集](https://x.com/m_olbap/status/1775201738397765775?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)**,以帮助文档AI的发展。

**5\. Misc Updates**:

*   **LLM训练和推理的效率突破**: Andrej Karpathy开源了[**llm.c**](https://github.com/karpathy/llm.c?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)，这是一个简洁的GPT-2训练实现，仅用1000行C/CUDA代码。这引发了将其移植到GPU以提高性能的讨论。Groq展示了成本效益的推理方法，而**4位量化**([HQQ](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the#L808))和**FP16xINT4内核**([Marlin](https://github.com/IST-DASLab/marlin?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the))等技术则承诺能带来速度提升。由Meta赞助的[语言模型物理研究](https://arxiv.org/abs/2404.05405?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)涉及了惊人的4.2M GPU小时.

*  检索增强生成(RAG)进展:RAG的创新包括使用[LlamaParse](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)提取文档知识图谱来增强高级工作流程,以及在[ARAGOG survey](https://twitter.com/llama_index/status/1777441831262818403?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)中对技术进行全面评估。多模态RAG正被应用于医疗领域,如[药丸识别](https://twitter.com/llama_index/status/1777722765589823728?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the),而即将到来的活动将展示[企业级RAG系统](https://twitter.com/llama_index/status/1777763272701468684?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)。

*   **框架探索和训练技术**: 像谷歌的**Griffin**这样的新颖架构凭借额外10亿个参数和更高的吞吐量超越了transformer。期待[**Jet MoE**](https://github.com/huggingface/transformers/pull/30005?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)整合到Hugging Face transformer中。对聊天模型的微调方法进行了仔细研究,将**直接偏好优化(DPO)**与**SFT+KTO**和微软的**DNO**等替代方法进行了对比。根据[PiSSA论文](https://arxiv.org/abs/2404.02948?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the),用SVD初始化**LoRA层**被发现可以显著提高微调结果。[最近的一项研究](https://arxiv.org/abs/2404.04125?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the)突出了多模态模型中零样本泛化的局限性。

* * *

