'Mergestral、Meta MTIAv2、Cohere Rerank 3、Google Infini-Attention'
============================================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-mergestral-meta-mtiav2-cohere-rerank-3/](https://buttondown.email/ainews/archive/ainews-mergestral-meta-mtiav2-cohere-rerank-3/) 

今天有一些小更新,均为值得关注之事,却未有明显的重大新闻。

* 新的8x22B Mixtral模型被[这位绝对疯狂的家伙](https://twitter.com/mejia_petit/status/1778390352082215129)_合并回到一个密集的模型_中,从而从8个专家中[提取出一个单独的专家](https://x.com/danielhanchen/status/1778453454375231553),有效地为我们提供了一个22B Mistral模型。
* Meta 宣布了他们的 [MTIAv2 chips](https://twitter.com/ylecun/status/1778392841083117939)，虽然你[无法购买或租赁](https://x.com/soumithchintala/status/1778107247022751822)这些芯片，但可以从远处欣赏它们。
* [Cohere Rerank 3](https://twitter.com/cohere/status/1778417650432971225)是一个用于[**增强**]企业搜索和RAG系统的基础模型。它能够准确检索100多种语言的多种结构化数据。[@aidangomez comment](https://twitter.com/aidangomez/status/1778416325628424339)。
* 一篇新的Google [关于Infini-attention的论文](https://twitter.com/swyx/status/1778553757762252863)展示了另一种超可扩展的线性注意力替代方案,这次展示了一个序列长度为1m的1B和8B模型。

与即将于下周开始推出的[Llama 3](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)相比,所有这些都是微不足道的。

* * *


AI Reddit Recap
===============


全新的模型与架构

*   **Mistral 8x22B**：已在 M2 Ultra 192GB 上运行，采用 4 位量化，在 M3 Max 128GB RAM 上可实现 [每秒 4.5 个 token 的出色性能](https://i.redd.it/skiryihkhqtc1.gif)。通过 [API](https://i.redd.it/eytf445jgntc1.png) 和 [基准测试](https://i.redd.it/2wnx1jjl8ptc1.jpeg) 展示。
*   **Command R+**: 这是[第一个在聊天机器人竞技场战胜GPT-4的开源模型](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus)，现已在HuggingChat上免费提供。该模型实现了[128k context length](https://www.reddit.com/r/LocalLLaMA/comments/1c0lkwo/how_does_command_r_achieve_128k_context/)，超越了其他大语境模型。
*   **MTIA芯片**：Meta宣布其[下一代训练和推理加速器](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)，具有改进的架构、密集计算性能、增加的内存容量和带宽。该产品旨在与PyTorch 2.0完全集成。
*   **UniFL**：[通过统一反馈学习提升了Stable Diffusion的性能](https://www.reddit.com/gallery/1c0qsz8)，在4步推理中的效果优于LCM和SDXL Turbo分别57%和20%。
*   **Infini-attention**：实现了[efficient infinite context transformers](https://arxiv.org/abs/2404.07143)，能够使模型处理长程依赖关系。

Stable Diffusion [与] Image Generation

*   **ELLA SDXL 权重**: [已确认永不发布](https://www.reddit.com/r/StableDiffusion/comments/1c0ryb5/ella_sdxl_weights_confirmed_to_never_be_released/)，作者将出版优先于产品可用性。社区对此感到失望，正期待 SD3。
* **SD 1.5**：被某些用户视为["king"](https://v.redd.it/08py75mvyptc1)的仍然展示了令人印象深刻的结果。

*   **16通道VAEs**：Stable Diffusion训练实验证明颇有挑战性,模型难以达到SDv1.5的质量水平。社区讨论[扩散训练中潜在空间的影响](https://www.reddit.com/r/StableDiffusion/comments/1c15qyd/how_do_the_vaes_latent_channels_or_the_specific/)。
* **CosXL**：来自Stability AI 的新模型在[革新图像编辑](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/)方面显示出前景。该模型的演示可在 Hugging Face 平台获得。

基于检索增强的生成 (RAG) 和上下文处理

*   **RAG pipeline评估**：共享了实用指南，重点强调了[构建生产就绪系统的挑战](https://www.reddit.com/r/MachineLearning/comments/1c0ryvz/d_a_practical_guide_to_rag_pipeline_evaluation/)，尽管单一演示很容易实现。
* 使用 R2R、SentenceTransformers 和 ollama/Llama.cpp 进行部署的简明易懂的[教程](https://www.reddit.com/r/LocalLLaMA/comments/1c0vht1/easy_local_rag_sharing_our_stepbystep_tutorial/)。
*   **RAG 与大型上下文模型**: [Gemini 概述](https://www.reddit.com/r/LocalLLaMA/comments/1c0iea2/rag_vs_large_context_models_a_gemini_overview/)对这些方法进行了比较,探讨了其未来相关性及使用场景依赖性。

开源工作和本地部署

* **LocalAI**：发布 v2.12.3，新增[增强的全功能图像生成、Swagger API、OpenVINO支持以及社区驱动的改进](https://www.reddit.com/r/LocalLLaMA/comments/1c0niro/localai_212_aio_images_improvements_swagger_api/)。
* 用户分享了使用 [HP z620 和 ollama/anythingllm](https://www.reddit.com/r/OpenAI/comments/1c12lxh/my_journey_with_local_ai_so_far_any_tips/) 的心得,并就持久性和升级等方面寻求建议。
*   **Llama.cpp**：不再提供二进制文件,致使某些人的编译工作更加困难。社区讨论[挑战和替代方案](https://www.reddit.com/r/LocalLLaMA/comments/1c0gop8/why_is_llamacpp_no_longer_providing_binaries/)。
* **AMD GPUs with ROCm**：使用[AUTOMATIC1111 和 kohya_ss 通过 Docker](https://www.reddit.com/r/StableDiffusion/comments/1c15khf/using_amd_gpu_with_rocm_for_automatic1111_and/)部署的指南，解决了兼容性问题。

Prompt 工程和 Fine-Tuning

*   **微调的 Prompt-response 示例**: 用户寻求对需要遵循特定输出格式的数量的建议，[估计范围从 50 到 10,000](https://www.reddit.com/r/MachineLearning/comments/1c0jmst/how_many_promptresponses_examples_do_i_need_for/)。
* 使用更大的LLMs进行提示输入：特别在RAG框架中，为较小的模型生成更好的提示存在[潜力](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning)。

基准测试、比较和评估

* Cohere Command R+: 与Claude 3、Qwen 1.5 72B和GPT-4相比,用户在写作风格自然性方面略有[轻微失望](https://www.reddit.com/r/LocalLLaMA/comments/1c0txo8/mild_disappointment_in_cohere_command_r/), 但在lmsys聊天竞技场基准测试方面表现出色。
* **Intel Gaudi**：据报道，在 Large Language Model 训练中，其速度较 NVIDIA 的产品快 [**50%**]，且[成本更低](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning/)。
* 测试新方法：就[推荐数据集、模型大小和基准](https://www.reddit.com/r/MachineLearning/comments/1c0vh48/d_what_would_you_recommend_testing_new_general/)进行讨论,以说服社区新型架构/优化器的优势。


* * *

AI Twitter Recap
================


**LLM Developments**

*   **Mixtral-8x22B 发布**: [@MistralAI](https://twitter.com/MistralAI/status/1778016678154211410) 发布了 Mixtral-8x22B，这是一个**176B MoE 模型,拥有约 40B 个激活参数,上下文长度达 65k 个 token**,在 Apache 2.0 许可下发布。早期评估结果显示,在 MMLU 上的得分达到了**77.3%**,优于其他开源模型。[@_philschmid](https://twitter.com/_philschmid/status/1778051363554934874) [@awnihannun](https://twitter.com/awnihannun/status/1778054275152937130)
*   **GPT-4 Turbo 改进** : 新版 GPT-4 Turbo 在编码基准测试方面表现出显著提升,在大多数任务上均优于 Claude 3 Sonnet 和 Mistral Large。[@gdb](https://twitter.com/gdb/status/1778071427809431789) [@gdb](https://twitter.com/gdb/status/1778126026532372486) [@bindureddy](https://twitter.com/bindureddy/status/1778108344051572746)
*   **Command R+ Release**: [@cohere](https://twitter.com/cohere/status/1778417650432971225)发布了Command R+，这是一个新的开放词汇模型，拥有强大的多语言功能,在某些非英语基准测试中表现优于GPT-4 Turbo。该模型具有高效的tokenizer,可以实现更快的推理和更低的成本。[@seb\_ruder](https://twitter.com/seb_ruder/status/1778385359660867744)、[@aidangomez](https://twitter.com/aidangomez/status/1778391705663729977)
* **Gemini 1.5 Pro**: Google发布了Gemini 1.5 Pro, [**增添**]了音频和视频输入支持。它现已通过API在180多个国家和地区提供使用。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321)

**Efficient LLMs**

* **Infini-attention for Infinite Context**：Google 引入了 Infini-attention，这是一种高效的方法,可将 Transformer 大型语言模型(LLMs)扩展至**内存和计算资源有限的无穷长输入**。它将压缩式内存纳入注意力机制,并构建了局部和长期注意力机制。[@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778230430090592454) [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778234586599727285)
* 将 LLaMA Decoder 应用于视觉任务：此项工作研究了将仅含解码器的 LLaMA 模型应用于视觉任务。直接应用因果掩码会导致注意力崩溃，因此他们 **重新定位类别标记并使用软掩码策略**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778237179740688845)
*   **llm.c**：[@karpathy](https://twitter.com/karpathy/status/1778153659106533806)发布了llm.c，这是一个**约1000行代码的GPT-2 C实现, 直接调用CUDA内核**。虽然比PyTorch的灵活性稍低且速度略慢, 但它提供了一个简单、精简的核心算法实现。[@karpathy](https://twitter.com/karpathy/status/1778128793166856368) [@karpathy](https://twitter.com/karpathy/status/1778135672420966788)

**Robotics and Embodied AI**

*   **敏捷足球技能学习**: DeepMind 利用强化学习训练 AI 代理展示[**了**]**如转身、踢球和追球等敏捷足球技能**。这些策略可以迁移到真实机器人上，并组合在一起来进行射门和封堵。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778377999202541642)
*   **OpenEQA 基准测试**:Meta 发布了 OpenEQA,这是一个旨在通过开放词汇问题来衡量嵌入式 AI 代理对物理环境理解能力的基准测试。当前的视觉语言模型 **在空间理解等方面远远低于人类水平**。[@AIatMeta](https://twitter.com/AIatMeta/status/1778425321118732578) [@AIatMeta](https://twitter.com/AIatMeta/status/1778425322645422396)

**Hardware and Systems**

* **MTIAv2 推理芯片**：Meta 宣布推出第二代推理芯片 MTIAv2，**采用 TSMC 5nm 工艺制造，提供 708 TFLOPS int8 性能**。该芯片使用标准的 PyTorch 堆栈以保持灵活性，并针对 Meta 的 AI 工作负载进行了优化。[@ylecun](https://twitter.com/ylecun/status/1778392841083117939) [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502) [@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854)

**Miscellaneous**

*   **Rerank 3 发布**：[@cohere](https://twitter.com/cohere/status/1778417650432971225) 发布了 Rerank 3，这是一个用于**增强企业搜索和 RAG 系统**的基础模型。它可精确检索100多种语言的多方面和半结构化数据。[@aidangomez](https://twitter.com/aidangomez/status/1778416325628424339)
*   **Zephyr对齐**: 使用包含 7k 个偏好比较的数据集，采用**Odds Ratio Preference Optimization (ORPO)** 训练了一个新的 Zephyr 模型，在 IFEval 和 BBH 上获得了高分。代码已在《Alignment Handbook》中开源。[@osanseviero](https://twitter.com/osanseviero/status/1778430866718421198) [@\_lewtun](https://twitter.com/osanseviero/status/1778430868387778677)
* Suno Explore 已经发布: [@suno_ai_](https://twitter.com/suno_ai_/status/1778430403973447708) 推出了 Suno Explore，这是一个**通过其 AI 系统生成新音乐流派的聆听体验**。
* **Udio 文本转音乐**：来自 Uncharted Labs 的全新 Udio 文本转音乐 AI 可以**从文本描述生成多种风格的完整歌曲**。早期演示效果非常出色。[@udiomusic](https://twitter.com/udiomusic/status/1778049129337192888)

* * *

AI Discord Recap
================


* 新 AI 模型的期待正在升温。AI 界正期待多个新模型的发布,包括预计在未来 1-3 周内推出的 Stability.ai 的 **SD3**,以及 Meta 确认即将推出的 **Llama 3**([TechCrunch 文章](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)),还有 MistralAI 的 **Mixtral-8x22b** 的指令微调版本。此外,Sophia Yang 也透露了一个全新的 Apache 2.0 许可的模型,在初步 [AGIEval 结果](https://x.com/jphme/status/1778030213881909451)中表现优于其他开源基础模型,引起了广泛关注。

*   **Mixtral模型在性能方面令人瞩目**: 最新推出的**Mixtral-8x22b**正引起轰动,根据[AGIEval结果](https://x.com/jphme/status/1778030213881909451),在PIQA和BoolQ等基准测试中明显超越其他开源模型。讨论也突出了即使经过量化,**Mixtral 8x7b**模型的出色表现。社区正在分析这些模型的功能,并[将其与GPT-4及其他领先系统进行对比](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4)。

*   **CUDA性能优化与量化**：在CUDA MODE Discord中，一位用户报告实现了GPT-2 110ms/迭代的纯CUDA正向传播，优于PyTorch。正在探索利用CUDA的C子集、内联汇编和协作组进行优化。HQQ（Half-Quadratic Quantization，半二次量化）社区正在研究量化脚本、int4内核的性能以及困惑分数的差异，最新的[HQQ代码已分享在GitHub上](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py)。

* 可访问的新 AI 应用程序和集成: 多个全新的 AI 应用程序和集成已被宣布推出,包括配有 GPT-4 和 Vision AI 功能的 **GPT AI**、提供免费高级模型 API 的 **Galaxy AI**、支持直观应用构建的 **Appstorm v1.6.0**,以及 **Perplexity AI 和 Raycast** 之间的合作,为 Raycast 订阅用户提供免费的 Perplexity Pro（[Raycast 博客文章](https://www.raycast.com/blog/more-ai-models)）。OpenAI 也已达到 1 亿 ChatGPT 用户,正过渡至预付费信用系统。

*   **AI硬件和基础设施的进步**:Meta推出了其**Meta Training and Inference Accelerator (MTIA)**,该加速器拥有354 TFLOPS/s (INT8)的性能,功耗仅90W,专为AI工作负载而设计([Meta博客文章](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/))。Intel即将推出的**Lunar Lake CPUs**将配备45 TOPS NPU,以在本地运行Microsoft的Copilot AI。芯片设计商和制造商(如TSMC)之间的供应链动态备受关注。

* * *

