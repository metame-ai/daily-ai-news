OpenAI's Instruction Hierarchy
========================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-openai-reveals-its-instruction-hierarchy/](https://buttondown.email/ainews/archive/ainews-openai-reveals-its-instruction-hierarchy/) 

总的来说，每个现代操作系统都具有"保护环(Protection Rings)"的概念，根据需要提供不同级别的权限。

![image.png](https://assets.buttondown.email/images/ef0283c2-2c8a-4aaf-84f0-11a7991c3b89.png?w=960&fit=max)

直到 ChatGPT 的出现，被训练为"过于激进的自动完成"的模型一直容易遭受 prompt 注入的影响。

![image.png](https://assets.buttondown.email/images/a6edde4d-e948-49ca-bf4b-7f93dcbebc61.png?w=960&fit=max)

因此，为 LLMs 设置特权级别无疑是解决之道。OpenAI [发表了一篇论文](https://arxiv.org/abs/2404.13208)，首次阐述了他们对此的看法。

![image.png](https://assets.buttondown.email/images/f2448e39-c8be-4591-a14a-daa0423b61b4.png?w=960&fit=max)

这被呈现为一个**对齐**问题 - 每个层级都可能**对齐**或**未对齐**,对未对齐的反应可以是**忽略并继续**或**拒绝**（如果无法继续）。作者综合数据以生成复杂请求的分解,放置于不同层级,根据对齐和注入攻击类型进行变化,应用于不同领域。


该结果为对所有提示注入进行建模的通用系统设计。如果我们可以为其生成数据, 就可以对其进行建模。

![image.png](https://assets.buttondown.email/images/14244827-5aa4-48f8-a0e2-456abdc84b99.png?w=960&fit=max)

使用该方法，他们几乎可以解决提示泄露问题，并将防御能力提高20-30个百分点。

作为一项有趣的附加功能，作者发现仅在系统提示中添加指令层次结构会降低基线 Large Language Model (LLM) 的性能，但通常会提升经过层次训练的 LLM 的性能。

![image.png](https://assets.buttondown.email/images/86c6e8f3-07dd-4cba-bbb8-457a54de88aa.png?w=960&fit=max)

* * *


AI Reddit Recap
===============


**AI Models and Benchmarks**


* Microsoft 发布了Phi-3 mini模型:在 /r/MachineLearning 上,Microsoft在Hugging Face平台上发布了轻量级的Phi-3-mini模型,[其基准测试数据令人瞩目,但需要第三方验证](https://www.reddit.com/r/MachineLearning/comments/1cb7f9n/n_phi3mini_released_on_huggingface/)。该模型提供了4K和128K两种上下文长度的版本。

* Apple发布了OpenELM高效语言模型系列。Apple在Hugging Face上开源了OpenELM语言模型系列，并提供了[开放的训练和推理框架](https://huggingface.co/apple/OpenELM)。该270M参数模型在MMLU上的表现优于3B参数模型,这表明这些模型存在欠拟合问题。该许可允许修改和再分发。

* 在 /r/LocalLLaMA 上，一个业余基准测试[**对12个模型的指令遵循能力**](https://www.reddit.com/r/LocalLLaMA/comments/1cbhsnc/instruction_accuracy_benchmark_12_models_tested/)进行了测试评估，涵盖了27个类别。其中，Claude 3 Opus、GPT-4 Turbo和GPT-3.5 Turbo位列前三甲，而Llama 3 70B超越了GPT-3.5 Turbo。

* Rho-1 方法利用仅 3% 的 token 即可训练出 SOTA 模型。在 /r/LocalLLaMA 中, Rho-1 方法[**仅使用 3% 的预训练 token 就可以达到 DeepSeekMath 的性能**](https://www.reddit.com/r/LocalLLaMA/comments/1cb4wr7/rho1_not_all_tokens_are_what_you_need_a_very/)。该方法采用参考模型在每个 token 层面对训练数据进行过滤, 同时还能够通过少量额外训练提升现有模型如 Mistral 的性能。

**AI Applications and Use Cases**

* Wendy's正在[**推出一款由AI驱动的自助取餐窗口点餐系统**](https://v.redd.it/h6yzjwx3g9wc1)。评论指出这可能为非英语母语者提供更佳体验,但也引发了对入门级工作影响的担忧。

* Gen Z 员工更倾向于向 AI 而非人类经理咨询职业建议：一项新研究发现, [**Gen Z 员工正选择从生成式 AI 工具而非他们的人类经理那里获取职业建议**](https://www.computerworld.com/article/2094650/gen-z-workers-pick-genai-over-managers-for-career-advice.html)。

* 生产环境中部署 Llama 3 模型：在 /r/MachineLearning 上有一个教程介绍了[**在 AWS EC2 实例上部署 Llama 3 模型**](https://www.reddit.com/r/MachineLearning/comments/1cb3ge1/d_how_to_and_deploy_llama_3_into_production_and/)。Llama 3 8B 需要 16GB 磁盘空间和 20GB VRAM，而 70B 则需要 140GB 磁盘和 160GB VRAM (FP16)。使用 vLLM 等推理服务器可将大模型拆分到多个 GPU 上运行。

* 从无表情面部预测政治取向的AI系统: 一项新研究声称，一个AI系统能够[仅通过分析无表情的面部照片就预测出人们的政治取向](https://www.psypost.org/artificial-intelligence-can-predict-political-beliefs-from-expressionless-faces/)。评论者持有怀疑态度,认为仅凭基本的人口统计学因素就可做出合理猜测,无需先进的AI技术。

* 在有一些提示的情况下，Llama 3 擅长创意写作。在 /r/LocalLLaMA 上, 一位业余作家发现 Llama 3 70B 是[非常出色的写罗曼史小说的创作伙伴](https://www.reddit.com/r/LocalLLaMA/comments/1cbrt5l/llama_3_70b_is_really_good_with_creative_writing/)。只需一两句示例写作和基本指令, 它就能生成有用的想法和段落, 然后作者对其进行修改和融入。

**AI Research and Techniques**

* HiDiffusion 技术使 Stable Diffusion 模型能够[**仅添加一行代码即可生成 2K/4K 分辨率的图像**](https://hidiffusion.github.io/),从而实现更高分辨率的图像生成。相比基础 SD,该技术提高了分辨率和生成速度。

* 基于进化模型合并技术,可以帮助开源项目与之竞争：随着计算能力成为庞大开放模型的瓶颈,[**模型合并、扩展和协同Transformer等技术可以帮助开源社区保持步伐**](https://i.redd.it/xcpvjcscrbwc1.jpeg)。

* 在/r/MachineLearning中提出了Gated Long-Term Memory(GLTM)单元作为[**高效的LSTM替代方案**](https://www.reddit.com/r/MachineLearning/comments/1caywsz/d_gated_longterm_memory/)。GLTM与LSTM不同的是，它并行执行"繁重的工作"，仅进行顺序的乘法和加法操作。GLTM使用线性而非二次内存，旨在成为高效的LSTM替代方案。

* * *

AI Twitter Recap
================


**AI Models and Architectures**

*   **Llama 3 Model**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1783013591714873765)指出Llama 3在回答3年级学生都能答出的问题时出现错误, 表示它不应被视为超人般的天才。[@bindureddy](https://twitter.com/bindureddy/status/1783150111364878508)建议使用Llama-3-70b进行推理和代码编写, 使用Llama-3-8b进行快速推理和微调。[@winglian](https://twitter.com/winglian/status/1783122644579090600)发现Llama 3在将rope\_theta设置为16M时, 可在65k词上下文中获得良好的召回率, [@winglian](https://twitter.com/winglian/status/1783013020412551200)还指出将rope\_theta设置为8M可在无需继续预训练的情况下, 在高达40K词上下文的深度上实现100%的passkey检索。
*   **Phi-3模型**: [@bindureddy](https://twitter.com/bindureddy/status/1782839198044811595)提出,如果Llama-3性能相当且成本低10倍,何必使用OpenAI的API。微软发布了Phi-3系列开放模型,包括mini (3.8B)、small (7B)和medium (14B)三种规模。据[@rasbt](https://twitter.com/rasbt/status/1782772068754325656)和[@_philschmid](https://twitter.com/_philschmid/status/1782781516172431685)称,**Phi-3-mini的性能可与Llama 3 8B匹敌**。[@rasbt](https://twitter.com/rasbt/status/1782778273895731213)指出,Phi-3 mini可量化到4位运行于手机。

*   **Snowflake Arctic**: [@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1783123091104936060)宣布了Snowflake Arctic,这是一款**480B参数的Dense-MoE LLM,专为企业用例而设计,如代码、SQL、推理和遵循指令**。[@_philschmid](https://twitter.com/_philschmid/status/1783140561483960620)指出它是在Apache 2.0下开源的。
* **Apple OpenELM**：Apple 发布了 OpenELM，这是一个高效的开源 LM 家族，根据 [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1782948858005454997) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1782949384163876953) 的说法，其性能与 OLMo 不相上下，但所需的预训练 token 数量只有后者的一半。
*   **Meta RA-DIT**: Meta研究人员开发了RA-DIT，这是一种微调方法,该方法根据[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1782907305748578360)的总结,利用检索增强型生成(RAG)的方式提升LLM的性能。

**AI Companies and Funding**

*   **Perplexity AI 融资**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782784338238873769) 宣布 Perplexity AI 已获得 6,270 万美元融资,公司估值达 10.4 亿美元,由 Daniel Gross 领投,还有 Stan Druckenmiller、NVIDIA、Jeff Bezos 等投资者参与。[@perplexity\_ai](https://twitter.com/perplexity_ai/status/1782782211399279076) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782785205419544731) 表示,这笔融资将用于扩大 Perplexity AI 在消费者和企业领域的应用覆盖范围。
*   **Perplexity Enterprise Pro**：Perplexity AI 推出了 Perplexity Enterprise Pro，这是一款企业级 AI 回答引擎，具有 **[增强的] 数据隐私、SOC2 合规性、SSO 和用户管理**功能，每月每个席位收费 40 美元，据 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782778575449661768) 和 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1782774382399557633) 报道。该产品已被 Databricks、Stripe、Zoom 等跨行业公司采用。
*   **Meta Horizon OS**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1782826465207165288)讨论了Meta为VR头戴设备推出的Horizon OS,指出其可以支持专用头戴设备和应用程序,但会拖慢Meta的软件开发进程。他认为**仅允许合作伙伴访问标准Quest硬件的完整OS,可以开放更多使用场景,同时成本较低**。

**AI Research and Techniques**

*   **指令层级**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1782878279504191896)重点强调了OpenAI关于"指令层级"的研究,认为系统提示比用户指令更为重要,以防止监狱破坏攻击。该研究鼓励模型从系统提示的视角来解释用户指令。
* Anthropic Sleeper Agent Detection: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1782908989296046210)发布了关于使用探测(probing)来检测当被破坏的"间谍(sleeper agent)"模型在训练中伪装成安全后即将表现出危险行为的研究。探测追踪模型在回答"是"与"否"安全问题时内部状态的变化。
* 根据 [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1782945719747510622) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1782952067339858036) 的介绍，Microsoft 推出了 Multi-Head Mixture-of-Experts (MH-MoE)。该系统将 tokens 拆分为分配给不同专家的子 tokens，以提高相比基准 MoE 的性能。
* SnapKV是一种方法,通过根据[@_akhaliq](https://twitter.com/_akhaliq/status/1782946902952034546)的自动压缩KV缓存来**高效地最小化LLMs中的KV缓存大小,同时保持性能**。该方法实现了3.6倍的加速和8.2倍的内存利用率改善。

* * *

AI Discord Recap
================


**1\. New AI Model Releases and Benchmarking**

*   **[Llama 3](https://huggingface.co/blog/llama3)** 已发布, 在 15 万亿个 tokens 上进行了训练, 并在 1000 万个人工标注的样本上进行了微调。 **70B 版本** 在 **MMLU** 基准测试中超越了公开的大型语言模型, 得分超过 80 分。 它采用了 **SFT、PPO、DPO 对齐**, 并使用了 **基于 Tiktoken 的 tokenizer**。 [demo](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct)

* 微软发布了[Phi-3 mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)（3.8B）和128k版本，利用**SFT & DPO**在3.3T个token上进行了训练。它在任务(如RAG和基于[LlamaIndex基准测试](https://twitter.com/llama_index/status/1782870458121282003)的路由任务)上与**Llama 3 8B**相匹配。[在本地运行](https://twitter.com/llama_index/status/1782893301214986593)

* **[Internist.ai 7b](https://huggingface.co/internistai/base-7b-v0.2)**，一款医疗领域的大型语言模型(Large Language Model, LLM)，在经过 **10 名医生的盲测评估**时表现优于 GPT-3.5，并且超过了美国医学入学考试(USMLE)合格分数，突显了 **数据策划** 和 **医生参与训练** 的重要性。

* 据@DingBannu和@testingcatalog在推特上发布的信息,业界对即将于4月29-30日推出的**新GPT**和**Google Gemini**发布情况充满期待。

**2\. Efficient Inference and Quantization Techniques**

*   **[Fireworks AI](https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs)** 讨论了通过将模型量化至 **FP8** 而无任何性能损失, 从而可实现比普通 LLM 快 4 倍的模型服务。Microsoft 的 **[BitBLAS](https://github.com/microsoft/BitBLAS)** 为量化 LLM 部署提供了支持混合精度矩阵乘法的功能。

*   **FP8**性能与**BF16**相比分别为29.5ms和43ms，尽管**Amdahl's Law**限制了收益。实现**批量大小上的确定性损失**是重点考虑因素，包括**CUBLAS_PEDANTIC_MATH**设置。

* CUDA kernels 在 llm.c 中就其在优化方面的潜在**教育价值**进行了讨论,并提出将其作为课程材料包括进去,突出 FP32 路径的可读性。

**3\. RAG Systems, Multi-Modal Models, and Diffusion Advancements**

* **[CRAG - Corrective RAG](https://twitter.com/llama_index/status/1782799757376963006)** 在 RAG 中添加了一个反馈层,用于将检索到的信息划分为"正确"、"错误"和"模糊"三类,以提高上下文感知。

*   **[Haystack LLM](https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb)** 现已根据 OpenAPI 规范建立索引,并根据意图检索出优质服务。**[llm-swarm](https://github.com/huggingface/llm-swarm)** 支持 LLM 推理的可扩展性。

* Adobe 发布了[Firefly Image 3](https://www.adobe.com/products/firefly.html)，以提升图像生成质量和控制。[HiDiffusion](https://github.com/megvii-research/HiDiffusion)使用"单行代码"提高了扩散模型的分辨率和速度。

* **[Multi-Head MoE](https://arxiv.org/abs/2404.15045)** 通过借用multi-head机制来提升专家激活和语义分析能力,相较于稀疏的MoE模型有所改进。

**4\. Prompt Engineering and LLM Control Techniques**

* 有关**prompt engineering**最佳实践的讨论,如利用**positive examples**引导风格,而非负面指令。神奇的**RageGPTee**开创了诸如**step-by-step**和**chain of thought** prompt等技术。

* 一篇关于[Self-Supervised Alignment with Mutual Information (SAMI)](https://arxiv.org/abs/2404.14313)的论文,无需偏好标签或演示即可对LLM进行微调,从而实现期望的效果,提高跨任务的性能。

* **[Align Your Steps](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps)** 是 NVIDIA 优化扩散模型采样策略的方法,旨在提高跨数据集输出的速度和质量。

* 对于 **LLM control theory** 的探索,例如使用 **greedy coordinate search** 比蛮力方法更高效地处理对抗输入[arXiv:2310.04444](https://arxiv.org/abs/2310.04444).

* * *

