Llama-3-70b 是 GPT-4 级别的开放模型。
================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-llama-3/](https://buttondown.email/ainews/archive/ainews-llama-3/) 

对于样本量为1600票的初步结果而言，Llama-3-70b在Lmsys的表现竟然比公布的基准还要出色,这在当下实属罕见。

![image.png](https://assets.buttondown.email/images/ebd4a52b-05e1-4b91-a842-0432853455fa.png?w=960&fit=max)

这是第一个打败Opus的开放模型,而Opus本身是第一个短暂打败GPT4 Turbo的模型。当然,这种情况可能会随时间而发生变化,但就Llama-3-400b而言,前景看好。

[Groq正在以500-800 tok/s的速度为70b模型提供服务](https://twitter.com/mattshumer_/status/1781355430914015482)，这使得Llama 3毋庸置疑地成为最快的GPT-4级别的模型。

最近关于 Chinchilla [复现结果](https://twitter.com/tamaybes/status/1780639257389904013) 的审查受到一些关注(不要错过 [Susan Zhang 的精彩文章](https://twitter.com/suchenzang/status/1616752482226671620)，已被 [Chinchilla 的合著者](https://twitter.com/borgeaud_s/status/1780988694163321250)所承认)。Llama 2 和 3(以及 Mistral)已基本确定将 Chinchilla 定律归入历史的垃圾堆。

* * *


AI Reddit Recap
===============


**Meta's Llama 3 Release and Capabilities**

*   **Llama 3发布为最强大的开放式 LLM**：Meta发布了[Llama 3，这是迄今为止最强大的可公开获取的大型语言模型](https://ai.meta.com/blog/meta-llama-3/)。在/r/LocalLLaMA中,已注意到[**拥有8B和70B参数版本,支持8K上下文长度**](https://www.reddit.com/r/LocalLLaMA/comments/1c7kd9l/llama_3_postrelease_megathread_discussion_and/)。还共享了一个开源的[70B模型代码解释器](https://github.com/e2b-dev/e2b-cookbook/blob/main/examples/llama-3-code-interpreter/llama_3_code_interpreter.ipynb)。

*   **Llama 3在基准测试中的表现超越前代模型**: 在 /r/LocalLLaMA 分享的基准测试中显示, [**Llama 3 8B instruct 模型在各类任务中的表现优于前代的Llama 2 70B instruct 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1c7kd9l/llama_3_postrelease_megathread_discussion_and/)。[基于API定价,70B模型的性能达到了GPT-4级别,但成本仅为其20分之一](https://www.reddit.com/r/LocalLLaMA/comments/1c7jybg/llama370b_over_20x_cheaper_than_gpt4/)。测试结果还表明, [Llama 3 7B在函数调用和算术方面的表现超越了Mistral 7B](https://www.reddit.com/r/LocalLLaMA/comments/1c7o27l/real_world_test_llama3_7b_blew_mistral_7b_out_of/)。

**Image/Video AI Progress and Stable Diffusion 3**

* 微软推出了[VASA-1用于从音频生成逼真的对话人脸](https://streamable.com/gzl8kr)。Meta的[图像和视频生成UI在/r/singularity中被誉为"令人难以置信"](https://www.reddit.com/r/singularity/comments/1c7hcvp/meta_ai_image_and_video_generation_ui_is/)。

*   **Stable Diffusion 3 印象和扩展**：在 /r/StableDiffusion 论坛上有人指出，[Imagine.art 对 SD3 功能的描述与其他服务相比存在一些误导性](https://www.reddit.com/r/StableDiffusion/comments/1c7p340/imagineart_gave_false_impression_of_sd3/)。同时还分享了一个名为 [Forge Couple 的扩展，可为 SD 添加可拖动的主体区域](https://www.reddit.com/r/StableDiffusion/comments/1c7lpd0/forge_couple_draggable_regions/)。

**AI Scaling Challenges and Compute Requirements**

*   **AI能源利用与GPU需求急剧增长**：在/r/singularity上的讨论中指出，[AI的计算需求可能在2030年前耗尽能源来源](https://www.reddit.com/r/singularity/comments/1c7282g/ais_voracious_need_for_computing_power_is/)。Elon Musk表示[训练Grok 3将需要100,000个Nvidia H100 GPU](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-says-the-next-generation-grok-3-model-will-require-100000-nvidia-h100-gpus-to-train)，而[AWS计划收购20,000个B200 GPU以建立一个27万亿参数的模型](https://www.cnbc.com/2024/03/18/nvidia-announces-gb200-blackwell-ai-chip-launching-later-this-year.html)。

**AI Safety, Bias and Societal Impact Discussions**

*   **政治偏向与AI安全担忧**：在 /r/singularity 上,有人认为[AI 所谓的"政治偏向"更多反映了政党立场,而非模型本身](https://www.reddit.com/r/singularity/comments/1c72lgh/the_political_bias_of_ai_model_is_an_indictment/)。Llama 3以其[在互动中的诚实和自我意识](https://www.reddit.com/r/LocalLLaMA/comments/1c7e1i0/llama_3_is_unquestionably_characterized_by_its/)为人所称道。讨论还涉及了[对有益AI发展的悲观论与乐观论](https://www.reddit.com/r/LocalLLaMA/comments/1c7e155/meta_llama3_pleasantly_surprised_to_be_provided/)的权衡。

*   AI[**有**]打破加密的潜力：在/r/singularity上的一篇帖子讨论了[ 量子密码破译 以及AI何时可能打破当前的加密方法](https://www.reddit.com/r/singularity/comments/1c7euhx/the_quantum_cryptopocalypse_how_soon_until/)。

* * *

AI Twitter Recap
================


**Meta Llama 3 Release**

*   **模型详情**: [@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440) 发布了 Llama 3 模型,尺寸为 **8B 和 70B**,还有一个 **400B+ 模型正在训练**。Llama 3 使用了 **128K 词汇 tokenizer**,训练数据量为 **15T tokens**(是 Llama 2 的 7 倍)。它有 **8K 的上下文窗口**,采用了 **SFT、PPO 和 DPO** 进行了对齐。
* 据[@karpathy](https://twitter.com/karpathy/status/1781028605709234613)所述，Llama 3 70B **广泛优于Gemini Pro 1.5和Claude 3 Sonnet**，Llama 3 8B **则优于Gemma 7B和Mistral 7B Instruct**。[@bindureddy](https://twitter.com/bindureddy/status/1780993893645132228)则强调，**400B版本在基准测试中的性能已接近GPT-4水平**。
*   **可用性**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1781068939641999388) 指出 Llama 3 是**从发布到在 Hugging Face 上成为第一大趋势的最快 AI 模型**。它也可以通过 [@awscloud](https://twitter.com/AIatMeta/status/1780997412418736591)、[@Azure](https://twitter.com/AIatMeta/status/1780997412418736591)、[@Databricks](https://twitter.com/AIatMeta/status/1780997412418736591)、[@GoogleCloud](https://twitter.com/AIatMeta/status/1780997412418736591)、[@IBM](https://twitter.com/AIatMeta/status/1780997412418736591)、[@NVIDIA](https://twitter.com/AIatMeta/status/1780997412418736591) 等渠道获得。

**Open Source AI Landscape**

*   **重要意义**: [@bindureddy](https://twitter.com/bindureddy/status/1781152808072626460)认为未来开源生态系统中的大部分AI创新将发生在Llama架构上。[@Teknium1](https://twitter.com/Teknium1/status/1781345814633390579)认为Llama 3 证实了**微调能够教会模型新知识**，以及10K样本是最佳指令微调数量的说法是错误的。
*   **计算趋势**：[@karpathy](https://twitter.com/karpathy/status/1781387674978533427)分享了关于**llm.c**的更新,该项目可在2K行C/CUDA代码中以与PyTorch相匹配的速度在GPU上训练**GPT-2**。他强调了**超优化代码**以提高性能的重要性。
*   **商业化**：[@abacaj](https://twitter.com/abacaj/status/1781443464246559180)认为 **Token价格正在暴跌**，因为任何人都能获取Llama权重并优化运行时间。[@DrJimFan](https://twitter.com/DrJimFan/status/1781386105734185309)预测**GPT-5将在Llama 3 400B发布之前宣布**，因为OpenAI根据开源进展决定发布时间。

**Ethical and Societal Implications**

* **员工待遇**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1781138214713274714)表达了对因参与抗议而被解雇的Googlers的同情, 指出即便存在分歧, 也需要尊重员工的重要性。

*   **数据透明化**: [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1781301149535981597)认为 **训练数据的透明化无疑能为社会带来积极影响**, 但目前公司缺乏这方面的激励措施。
*   **伦理要求**: [@francoisfleuret](https://twitter.com/francoisfleuret/status/1781290862002962778)构想了一个世界,在该世界中,**电子邮件和 Web 客户端必须遵守与当前 LLM 相同的伦理要求**.

* * *

AI Discord Recap
================


**Meta's Llama 3 Release Sparks Excitement and Debate**

*   **Meta发布了Llama 3**，这是一个全新的**大型语言模型**系列，涵盖了从**8B到70B参数**的不同规模版本，包括针对对话优化的预训练和指令微调版本。Llama 3 采用了全新的**128k token tokenizer**，支持多语言使用，并声称相比之前的模型具有更出色的推理能力。[博客](https://ai.meta.com/blog/meta-llama-3/)

*   讨论围绕着 Llama 3 的性能基准测试与 GPT-4、Mistral 和 GPT-3.5 等模型展开。有人赞扬其[**人类般的**]回应,而另一些人则指出尽管其经过多语种训练,但在非英语语言中仍存在局限性。

*   对于 Llama 3输出的下游使用的 **Licensing restrictions** 受到一些人的批评,认为其阻碍了开源开发。[Tweet](https://fxtwitter.com/xlr8harder/status/1780992684062024138)

* 围绕 Meta 计划推出的**Llama 3模型(405B参数)**的期待正在逐步升温,该模型有望开放权重,并可能改变开源AI与封闭模型(如GPT-5)的格局。

* 在 Llama 3 跨平台集成过程中,**Tokenizer 配置问题**、**无限响应循环**以及与现有工具(如 LLamaFile)的**兼容性**等问题都得到了讨论。

**Mixtral Raises the Bar for Open-Source AI**

* Mistral AI公司推出的**Mixtral 8x22B**模型被称赞为在开源AI领域树立了**性能和效率**的新标准。该模型采用了稀疏的**Mixture-of-Experts (MoE)架构**。[[YouTube](https://www.youtube.com/watch?v=N8U6XnVK2mM)]

* 基准测试结果显示，尽管**Mera-mix-4x7B MoE模型**较Mixtral 8x7B更小巧，但在**OpenLLM Eval**评测中取得了75.91的出色成绩，展现了其出色的性能。

*   已对**多语言能力**进行了探索,采用新的**Mixtral-8x22B-v0.1-Instruct-sft-en-de**模型在英语和德语数据上进行了fine-tuning。

* 在大型模型训练过程中,讨论了一些技术挑战,如**Shape Errors**、**OOM Issues**以及调优**router_aux_loss_coef**参数等问题。

**Efficient Inference and Model Compression Gain Traction**

* 由**Unsloth AI**提出的**量化技术**,如GPTQ和**4位模型**旨在提高大型模型的推理效率,与标准实现相比,可减少**高达80%的内存使用**。

*   **LoRA (Low-Rank Adaptation)** 和 **Flash Attention** 被推荐用于高效进行 **LLM fine-tuning**,同时也有 DeepSpeed 等工具用于梯度检查点处理。

* 像 **Half-Quadratic Quantization (HQQ)** 和潜在的 **CUDA kernel 优化** 等创新方法被探索用于进一步压缩和加速 GPU 上的大型模型。

* 面向注重成本的开发者部署LLMs的**无服务器推理解决方案**,采用可负担的GPU托管方式进行分享。

**Open-Source Tooling and Applications Flourish**

*   **LlamaIndex** 展示了多个项目: 利用 Elasticsearch 构建 **RAG applications** [Blog](https://t.co/QqLdz5lojV)、支持 Llama 3 [Tweet](https://t.co/RMB7MhXIOA)，以及创建 **code-writing agents** [Collab](https://t.co/d6dHazOK93)。

*   **LangChain** 迎来了一个 **prompt Engineering** 课程的发布 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7186761950138109952/),同时也推出了利用旅行 API 的 **Tripplanner Bot** [GitHub](https://github.com/abhijitpal1247/TripplannerBot)。

* Cohere用户探讨了**数据库集成**、**RAG工作流程**以及边缘部署的**商业许可限制**。

*   **OpenRouter**已确认在Olympia.chat进行**生产使用**,并预计**与Llama 3实现整合**,同时**LM Studio**在v0.2.20版本中发布了对Llama 3的支持。

**Emerging Research Highlights**

* 一种新的**最佳适配打包算法**优化了用于LLM训练的文档打包,从而减少了截断 [论文](https://arxiv.org/abs/2404.10830)。

* **softmax瓶颈**与较小的LLMs中的饱和和性能不佳有关 [论文](https://arxiv.org/abs/2404.07647)。

* DeepMind 分享了关于可解释性的 Sparse Autoencoders (SAEs) 的进展[博客](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary)。

* **Chinchilla Scaling Laws**被重新解释,表明在最佳[scaling]过程中,更多参数可以优先于数据。

* * *

