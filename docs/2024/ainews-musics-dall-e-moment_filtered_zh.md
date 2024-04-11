Music的Dall-E时刻
================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-musics-dall-e-moment/](https://buttondown.email/ainews/archive/ainews-musics-dall-e-moment/) 

> 带链接版本: https://github.com/metame-ai/daily-ai-news/blob/main/docs/2024/ainews-musics-dall-e-moment_filtered_zh.md

当人们仍在消化昨天的 [Gemini audio](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)、[GPT4T](https://twitter.com/miramurati/status/1777834552238723108?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 和 [Mixtral](https://twitter.com/_philschmid/status/1778051363554934874?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 的重大新闻时,今天是 [Udio's big launch](https://twitter.com/udiomusic/status/1778045325468426431?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

![image.png](https://assets.buttondown.email/images/a8f8a3c9-d95a-4250-9f10-1f8ef80eaf7d.png?w=960&fit=max)

Suno当然有[自己的粉丝](https://twitter.com/tobi/status/1775684945257611286?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。Udio在最近几天[泄露相关消息](https://x.com/legit_rumors/status/1777059367788982389?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment),这并不令人惊讶,但更令人惊讶的是Sonauto _也_ 在今天发布,也进军音乐生成游戏,尽管远不那么完美。与Latent Diffusion不同的是,目前还不清楚是什么突破使Suno/Udio/Sonauto在同一时间出现。你可以在[Suno's Latent Space pod](https://www.latent.space/p/suno?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)上听到一些线索。

* * *


AI Reddit Recap
===============


**AI Models and Architectures**

*   **Google的Griffin架构优于transformers**：在/r/MachineLearning上，Google发布了一个采用新的Griffin架构的模型，该模型[**在MMLU和平均基准评分的测试中优于多种大小的transformers**](https://i.redd.it/triygw613htc1.jpeg?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。Griffin在长上下文中提供了更高的效率、更快的推理和更低的内存使用。
*   **Command R+ 攀升排行榜, 超越了GPT-4模型**: 在 /r/LocalLLaMA 中, Command R+ [**已经攀升至 LMSYS Chatbot Arena 排行榜的第6位, 成为最佳开放模型**](https://www.reddit.com/r/LocalLLaMA/comments/1bzo2sh/latest_lmsys_chatbot_arena_result_command_r_has/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。根据[排行榜结果](https://chat.lmsys.org/?leaderboard&utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment), 它战胜了GPT-4-0613和GPT-4-0314。
*   **Mistral 发布 8x22B 开源模型，上下文 64K**: Mistral AI 发布了[**8x22B模型，上下文窗口为 64K**](https://x.com/mistralai/status/1777869263778291896?s=46&utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。它总共有 130B 个参数，每次前向传播有 44B 个激活参数。
*   **Google开源基于Gemma架构的CodeGemma模型**: Google发布了[基于Gemma架构的开源CodeGemma模型](https://huggingface.co/blog/codegemma?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)，并上传了预量化的4位模型以实现4倍的下载速度。

**Open Source Efforts**

*   **Ella权重发布，针对Stable Diffusion 1.5**: 在/r/StableDiffusion中，[**开放Ella权重，通过LLM帮助扩散模型增强语义对齐**](https://github.com/TencentQQGYLab/ELLA?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。
*   **Unsloth release 使fine-tuning内存减少**: 在 /r/LocalLLaMA 中, Unsloth 使用 GPU 和系统 RAM 之间的异步卸载提供 [**4x 更大的 context windows 和 80% 内存减少**](https://www.reddit.com/r/LocalLLaMA/comments/1bzywjg/80_memory_reduction_4x_larger_context_finetuning/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。
*   **Andrej Karpathy 发布了纯C语言实现的LLMs**: 在 /r/LocalLLaMA 中, 纯C语言的实现[**可能达到更快的性能**](https://www.reddit.com/r/LocalLLaMA/comments/1bztawh/andrejs_llms_in_pure_c_potentially_making_things/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

**Benchmarks and Comparisons**

*   **Command R+ model在M2 Max MacBook上实时运行**：在/r/LocalLLaMA中，推理[**使用iMat q1量化实时运行**](https://v.redd.it/b5sn5at5mftc1?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。
Cohere的 Command R 模型在排行榜上表现良好：在 /r/LocalLLaMA 中， Command R [**相比竞争对手具有较低的 API 成本**](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)，同时在聊天机器人竞技场排行榜上表现出色。

**Multimodal AI**

*   **Gemini 1.5的音频功能令人印象深刻**：在/r/OpenAI中, Gemini 1.5可以[**从纯音频剪辑中识别语音tone并根据名称识别说话者**](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。
*   **多模态storytelling入门套件**: 在 /r/OpenAI 中, 该套件利用 VideoDB、ElevenLabs 和 GPT-4 来[**生成纪录片风格的旁白**](https://www.reddit.com/r/OpenAI/comments/1bzncf2/starter_kit_for_storytelling_using_multimodal/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

* * *

AI Twitter Recap
================


**GPT-4 Turbo Model Improvements**

*   **改进的推理和编码能力**：[@gdb](https://twitter.com/gdb/status/1778071427809431789?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)、[@polynoamial](https://twitter.com/polynoamial/status/1777809000345505801?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@BorisMPower](https://twitter.com/BorisMPower/status/1777867583947227582?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)注意到GPT-4 Turbo与先前版本相比大幅提高了推理和编码性能。
*   **完全可用**: [@gdb](https://twitter.com/gdb/status/1777776125139194252?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)、[@miramurati](https://twitter.com/miramurati/status/1777834552238723108?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@owencm](https://twitter.com/owencm/status/1777770827985150022?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)宣布 GPT-4 Turbo 现已结束预览,并全面上线。
*   **与以前版本的比较**: [@gdb](https://twitter.com/gdb/status/1778126026532372486?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)、[@nearcyan](https://twitter.com/nearcyan/status/1777893558072270889?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@AravSrinivas](https://twitter.com/AravSrinivas/status/1777837161040990356?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)分享了比较结果,并指出这次更新非常值得关注。

**Mistral AI's New 8x22B Model Release**

*   **176B parameter MoE model**: [@sophiamyang](https://twitter.com/sophiamyang/status/1777945947764297845?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 和 [@\_philschmid](https://twitter.com/_philschmid/status/1778051363554934874?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 详细介绍了 Mistral AI 发布的 Mixtral 8x22B，这是一个 176B parameter MoE model，具有 65K context length 和 Apache 2.0 license。
*   **评估结果**: [@_philschmid](https://twitter.com/_philschmid/status/1778083833507659997?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 分享 Mixtral 8x22B 在 **MMLU** 上获得了 **77%** 的成绩。更多积极的结果在 [@_philschmid](https://twitter.com/_philschmid/status/1778089353849290843?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 中。
*   **社区兴奋度和访问权限**: 很多人如 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1777904372091118026?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1777903886075875762?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 表示兴奋。它可以在Hugging Face和Perplexity AI上获得, 如 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1778117267005346286?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 所述。

**Google's New Model Releases and Announcements**

*   **Gemini 1.5 Pro公开预览**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777738279137222894?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)宣布Gemini 1.5 Pro已在Vertex AI上公开预览,具有长上下文窗口。根据[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment),可通过API在180多个国家使用。
*   **Imagen 2 更新**: Imagen 2 现在可以创建4秒的实时图像,并包括一个称为SynthID的水印工具,由[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747320945234422?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747324489306302?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)分享。
*   **CodeGemma和RecurrentGemma模型**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078071188304106?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 与Google Cloud合作,推出了用于编码的CodeGemma和用于提高内存效率的RecurrentGemma,详见[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078073377706083?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078075713982544?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

**Anthropic's Research on Model Persuasiveness**

*   **评估语言模型的说服力**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728366101119101?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 开发了一种测试说服力的方法,并分析了模型大小的扩展性。
*   **模型代际的缩放趋势**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728370148577657?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 发现较新的模型更有说服力。Claude 3 Opus 在统计上与人类论点相似。
*   **实验细节**: 在阅读了较不极端化问题上的 Language Model 或人类论点之后, 测量了意见一致水平的变化。在[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728378675536357?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728376960106587?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728375198568611?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)中有解释。

**Cohere's Command R+ Model Performance**

*   **Chatbot Arena上的顶级开放权重模型**: [@cohere](https://twitter.com/cohere/status/1778113095820526038?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)和[@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)庆祝Command R+在Chatbot Arena上达到第6名,与GPT-4并列为基于13K+投票的顶级开放模型。
*   **高效多语言分词**: [@seb\_ruder](https://twitter.com/seb_ruder/status/1778028863580188740?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)详细介绍了 Command R+ 的分词器较其他分词器能够将多语言文本压缩1.18-1.85倍更高效,从而实现更快的推理和更低的成本。
*   **访问和演示**: Command R+ 可在Cohere的playground (https://txt.cohere.ai/playground/) 和Hugging Face (https://huggingface.co/spaces/cohere/command-r-plus-demo) 上使用，来自 [@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 和 [@nickfrosst](https://twitter.com/nickfrosst/status/1777724060257968505?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

**Meta's New AI Infrastructure and Chip Announcements**

*   **下一代 MTIA 推理芯片**: [@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 宣布 MTIAv2，Meta 第二代推理芯片，采用 TSMC 5nm 工艺，具有 708 TF/s Int8 性能、256MB SRAM 和 128GB 内存。与第一代相比，计算密度提高 3.5 倍，稀疏计算性能提高 7 倍，如 [@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 所述。
*   **平衡计算、内存、带宽**: [@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 指出 MTIA 的体系结构优化了计算、内存带宽和容量平衡,用于排序和推荐模型。与 [@AIatMeta](https://twitter.com/AIatMeta/status/1778083241632604456?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 每个 GPU 相比,全栈控制可以随时间提高更大的效率。
*   **增长的AI基础设施投资**：Meta日益增加的AI基础设施投资的一部分,旨在推动新的体验,补充现有和未来的AI硬件,[@AIatMeta](https://twitter.com/AIatMeta/status/1778083243050275143?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)强调。


* * *

AI Discord Recap
================


**1) New and Upcoming AI Model Releases and Benchmarks**

*  关于 **[Mixtral 8x22B](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)** 的发布令人兴奋,这是一个拥有176B参数的模型,在诸如 **AGIEval** 等基准测试中表现优于其他开源模型。([tweet](https://x.com/jphme/status/1778030213881909451?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)) 一个 [磁力链接](https://x.com/MistralAI/status/1777869263778291896?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment) 已经被分享。

*  Google悄悄推出了[Griffin](https://huggingface.co/google/recurrentgemma-2b?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)，一个2B递归线性注意力模型（[paper](https://arxiv.org/abs/2402.19427?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)），以及新的代码模型[CodeGemma](https://huggingface.co/spaces/ysharma/CodeGemma?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)。

* OpenAI的**GPT-4 Turbo**模型已经发布,具有视觉能力、JSON模式和函数调用功能,相比于以前的版本具有显著的性能改进。讨论集中在其速度、推理能力以及构建高级应用程序的潜力。 ([OpenAI定价](https://openai.com/pricing?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)、[OpenAI官方推特](https://twitter.com/OpenAIDevs/status/1777769463258988634?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))。它在性能方面取得了显著的进步,并与**Sonnet**和**Haiku**等模型在[基准比较](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing&utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)中进行了讨论。

*   对于即将发布的 Llama 3、Cohere 和 Gemini 2.0 等产品,人们对其潜在影响产生了猜测和期待。

**2) Quantization, Efficiency, and Hardware Considerations**

*   关于提高效率的量化技术（如HQQ（[code](https://github.com/mobiusml/hqq?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)）和Marlin）的讨论,同时也关注维持困惑度。

*   Meta的研究关于**LLM知识容量规模化规律**（[论文](https://arxiv.org/abs/2404.05405?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)）发现**int8量化**能够利用高效的**MoE**模型保留知识。

*   **Hardware limitations** 对于在本地运行大型模型(如 Mixtral 8x22B)有限, 对于像 **multi-GPU support** 这样的解决方案感兴趣。

*   比较来自**Meta**、**Nvidia**和**Intel的Habana Gaudi3**等公司的**AI加速硬件**。

**3) Open-Source Developments and Community Engagement**

*   **LlamaIndex**用于**企业级检索增强型生成(RAG)**([blog](https://t.co/ZkhvlI4nnx?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))的应用展示,在ICLR 2024上利用MetaGPT框架联合RAG([link](https://t.co/sAF41j0uL4?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))。

*   新工具如**mergoo**用于**合并LLM专家**([GitHub](https://github.com/Leeroo-AI/mergoo?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))和**PiSSA**用于**LoRA层初始化**([paper](https://arxiv.org/abs/2404.02948?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment), [repo](https://github.com/GraphPKU/PiSSA?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)).

*   社区项目: everything-rag chatbot ([HuggingFace](https://huggingface.co/spaces/as-cle-bert/everything-rag?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment)), TinderGPT dating app ([GitHub](https://github.com/GregorD1A1/TinderGPT?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))。

*   社区成员在**HuggingFace**上快速开源新的模型,如Mixtral 8x22B。

**4) Prompt Engineering, Instruction Tuning, and Benchmarking Debates**

*   关于使用AI生成指令的**prompt engineering**策略，如**meta-prompting**和**iterative refinement**的广泛讨论。

* 比较**instruction tuning**方法: **RLHF** 与 **Direct Preference Optimization (DPO)** 在**StableLM 2**([model](https://huggingface.co/stabilityai/stablelm-2-12b-chat?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))中的使用。

*   对于被"gamed"的 **benchmarks** 持怀疑态度，建议使用人工评判的排行榜，如 **arena.lmsys.org**。

*   关于将LLMs用作**text encoders**的**LLM2Vec**的辩论([paper](https://arxiv.org/abs/2404.05961?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment), [repo](https://github.com/McGill-NLP/llm2vec?utm_source=ainews&utm_medium=email&utm_campaign=ainews-musics-dall-e-moment))及其实用价值。

* * *

