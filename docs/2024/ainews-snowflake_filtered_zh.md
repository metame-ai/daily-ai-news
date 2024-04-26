Snowflake: 128x4B MoE
=======================================================================

> 翻译转载自: [https://buttondown.email/ainews/archive/ainews-snowflake/](https://buttondown.email/ainews/archive/ainews-snowflake/) 

这项工作需要进一步分析验证,但也体现了 Snowflake 值得称道的努力,因为此前 Snowflake 在现代 AI 浪潮中较为低调。[Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/) 引人注目的原因,很可能并非其在页面上突出展示的令人困惑/难以联系的图表。

![image.png](https://assets.buttondown.email/images/8a45195d-2c7c-420b-a6cc-dcf124fc1d84.png?w=960&fit=max)

「企业智能」这一概念相当引人入胜,尤其是它能解释为何他们选择在某些领域表现优于其他领域的原因。

![image.png](https://assets.buttondown.email/images/1ab962bf-83de-4037-8fc1-b0a1e1bfa9d3.png?w=960&fit=max)

这张图表有趣之处在于，Snowflake基本上声称他们构建了一个在几乎每个方面都优于其数据仓库之战主要竞争对手Databricks的Transformer模型。(这肯定会让Jon Frankle和他那群快乐的Mosaics感到冒犯吧？)

下游用户并不非常关注训练效率,但另一个应该引起你注意的是模型架构 - 从 [DeepSeekMOE](https://x.com/deepseek_ai/status/1745304852211839163) 和 [DeepSpeedMOE](https://arxiv.org/pdf/2201.05596) 中吸取正确启示,专家越多 = 效果越佳。

![image.png](https://assets.buttondown.email/images/bcd39b75-ae22-43fa-be38-72ce278d1140.png?w=960&fit=max)

文中未提及DeepSeek使用过的"shared expert"技术。

最终提及了一个分为3个阶段的课程设置:

![image.png](https://assets.buttondown.email/images/21e13aaa-64f0-4924-b7d9-9c80f723e6ec.png?w=960&fit=max)

该做法与最近发表的Phi-3论文中提出的类似策略遥相呼应。

![image.png](https://assets.buttondown.email/images/24598097-2055-4691-89dd-90c83d91df37.png?w=960&fit=max)

最终，该 Model 以 Apache 2.0 许可证发布。

说实话这是一个很棒的发布,可能唯一不太理想的就是[Snowflake Arctic cookbook](https://medium.com/@snowflake_ai_research/snowflake-arctic-cookbook-series-exploring-mixture-of-experts-moe-c7d6b8f14d16)被发布在Medium.com上。

* * *


AI Reddit Recap
===============


AI Image/Video Generation

* 在 /r/StableDiffusion 中, Nvidia 推出的[Align Your Steps](https://www.reddit.com/gallery/1ccdt3x)可显著提升低步数下的图像质量, 使用更少的步数即可得到优质图像。该技术最适合搭配 DPM~ 采样器使用。
*   **Stable Diffusion 模型对比**：在 /r/StableDiffusion 上有一个[当前 Stable Diffusion 模型的大型对比](https://www.reddit.com/gallery/1ccetp2)，结果显示 SD Core 在手部/解剖学方面最优秀，而 SD3 对提示的理解能力最强但具有电子游戏风格。
*   **SD3 与 SD3-Turbo 对比**：基于 Llama-3-8b 语言模型生成的提示,涉及 AI、意识、自然和技术主题的 [8 张 Stable Diffusion 3 和 SD3 Turbo 模型生成的图像](https://www.reddit.com/r/StableDiffusion/comments/1ccj3kc/4_images_by_sd3_and_4_images_by_sd3turbo_prompts/)。

Other Image/Video AI

*   **Adobe AI 视频超分辨率**: [Adobe 的令人印象深刻的 AI 超分辨率项目](https://www.theverge.com/2024/4/24/24138979/adobe-videogigagan-ai-video-upscaling-project-blurry-hd)可以使模糊的视频呈现高清效果。然而，[在高分辨率下会出现更多失真和错误](https://v.redd.it/8pi8t62btewc1)。
* 在 /r/StableDiffusion 上, [Instagram 垃圾邮件发送者正在使用 FaceFusion/Roop](https://www.reddit.com/r/StableDiffusion/comments/1cbu5cx/how_are_these_instagram_spammers_getting_such/) 创造出逼真的人脸替换视频, 当视频中的人脸离镜头较远且分辨率较低时, 这种方法效果最佳。

Language Models and Chatbots


*   **Apple 开源 AI 模型**：[Apple 发布了端上语言模型的代码、训练日志及多个版本](https://www.macrumors.com/2024/04/24/apple-ai-open-source-models/)，这与通常只提供模型权重和推理代码的做法不同。
*   **L3 和 Phi 3 性能表现**: L3 70B 在 LMSYS 排行榜上的英语问题中[并列第一](https://i.redd.it/3fwedc7yqjwc1.png)。Phi 3 (4B 参数) [在香蕉逻辑基准测试中胜过 GPT 3.5 Turbo](https://i.redd.it/h6nvy99vjewc1.png) (~175B 参数)。
*   **Llama 3 推理和量化**: 一个[视频展示了Llama 3在MacBook上的快速推理](https://v.redd.it/qzg34xylgjwc1)。然而, [对Llama 3 8B进行量化](https://www.reddit.com/r/LocalLLaMA/comments/1cci5w6/quantizing_llama_3_8b_seems_more_harmful_compared/), 尤其是低于8位, 性能明显比其他模型有所下降。

AI Hardware and Infrastructure

*   **Nvidia DGX H200 for OpenAI**: [Nvidia CEO向OpenAI提供了一台DGX H200系统](https://i.redd.it/wnxzyyqurhwc1.jpeg)。一个[Nvidia AI数据中心](https://www.youtube.com/watch)如果全面建成,可在数分钟内训练出ChatGPT4,被描述为拥有"非凡的能力和复杂性"。
* 联想发布了[ThinkSystem AMD MI300X](https://lenovopress.lenovo.com/lp1943-thinksystem-amd-mi300x-192gb-750w-8-gpu-board)192GB 750W 8-GPU板卡的产品指南。

AI伦机和社会影响

AI Ethics and Societal Impact

*   **深度伪造裸露相关立法**：[位于两位数州的议员](https://www.nytimes.com/2024/04/22/technology/deepfake-ai-nudes-high-school-laws.html)正在起草或通过法案来打击针对未成年女孩的AI生成的色情图像, 这是由于青少年女孩受到影响所致。
*   **AI在政治中**: 在 /r/StableDiffusion 上，一个[奥地利政党使用AI生成了他们候选人相比真实照片更 男性化 的图片](https://www.reddit.com/gallery/1c7rikz)，这引发了使用AI在政治领域歪曲现实的隐患。
*   **AI对话保密性**：在/r/singularity上的一篇帖子论述，随着AI代理获得更多个人知识，[该关系应该拥有与医生和律师一样的法律保护保密性](https://www.reddit.com/r/singularity/comments/1cchgqs/ai_conversations_should_be_confidential_like/)，但公司很可能拥有并使用这些数据。

* * *

AI Twitter Recap
================


**OpenAI and NVIDIA Partnership**

*   **NVIDIA DGX H200交付给OpenAI**: [@gdb](https://twitter.com/gdb/status/1783234941842518414)指出NVIDIA亲自将世界上第一台DGX H200交付给OpenAI,**由Jensen Huang专门"为推进AI、计算机和人类"而捐赠**。 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783236039714189520)重点介绍了DGX GH200的特性, 如256个H100 GPU、1.3TB GPU内存和8PB/s的互联带宽。
*   **OpenAI和Moderna合作**: [@gdb](https://twitter.com/gdb/status/1783529202974687527)还提到了OpenAI和Moderna之间的合作,利用AI[**以加快**]药物发现和开发的步伐。

**Llama 3 and Phi 3 Models**

*   **Llama 3**: [@winglian](https://twitter.com/winglian/status/1783456379199484367)利用PoSE和RoPE theta调整, 将Llama 3 8B的上下文长度扩展至**96k**。 [@erhartford](https://twitter.com/erhartford/status/1783273948022755770)发布了Dolphin-2.9-Llama3-70b, 这是一个**经过微调的Llama 3 70B版本**, 由多人共同创建。 [@danielhanchen](https://twitter.com/danielhanchen/status/1783214287567347719)指出**Llama-3 70b QLoRA微调速度快1.83倍, VRAM占用量减少63%**, 而Llama-3 8b QLoRA可以适配8GB显存卡。
*   **Phi 3**: [@rasbt](https://twitter.com/rasbt/status/1783480053847736713)分享了有关 Apple 的 OpenELM 论文的详细信息,介绍了**4 种规模（2.7亿到30亿参数）的 Phi 3 模型族**。主要架构变化包括**采用来自 DeLighT 论文的逐层缩放策略**。实验结果表明,对于参数高效的微调, LoRA 和 DoRA 之间的性能差异并不明显。

**Snowflake Arctic Model**

*   Snowflake发布开源LLM: [@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1783123091104936060)宣布了Snowflake Arctic,这是一个**480B Dense-MoE模型,专为企业AI设计**。该模型将一个10B的Dense Transformer与一个128x3.66B的MoE MLP相结合。[@omarsar0](https://twitter.com/omarsar0/status/1783176059694821632)指出,该模型声称在达到编码、SQL和指令执行等类似于Llama 3 70B的企业指标的同时,计算需求仅为Llama 3 70B的1/17。

**Retrieval Augmented Generation (RAG) and Long Context**

*   **LLM中的检索头**: [@Francis_YAO_](https://twitter.com/Francis_YAO_/status/1783446286479286700)发现了检索头，这是一种特殊类型的注意力机制,负责LLM中的长上下文事实性。这些检索头具有通用性、稀疏性和因果性,并显著影响了链式思维推理。遮蔽这些检索头会使模型"失去"重要的先前信息。
* 高效 LLM 推理的 XC-Cache：[@_akhaliq](https://twitter.com/_akhaliq/status/1783554087574733294) 分享了一篇关于 XC-Cache 的论文,它可以缓存上下文以进行高效 LLM 生成,而非即时处理。该方法显示了可喜的加速和内存节省效果。
*   **RAG Hallucination Testing**: [@LangChainAI](https://twitter.com/LangChainAI/status/1783165455521481191)演示了如何使用 LangSmith 评估 RAG 管道并针对幻觉进行测试,方法是对输出内容与检索到的文档进行对比检查。

**AI Development Tools and Applications**

* 集成AI的CopilotKit：[@svpino](https://twitter.com/svpino/status/1783488942152528327)介绍了CopilotKit，这是一个开源库，可让将AI轻松集成到应用程序中，允许您将LangChain agents 引入您的应用程序、构建聊天机器人以及创建RAG工作流程。
* 用于LLM UX的Llama Index: [@llama_index](https://twitter.com/llama_index/status/1783297521386934351)展示了如何使用create-llama构建具有可扩展来源和引用的LLM聊天机器人/agent的UX。

**Industry News**

*   **Meta公司的AI投资**：[@bindureddy](https://twitter.com/bindureddy/status/1783296389671444521)指出,Meta第二季度的预测疲软,并计划在AI上投入数十亿美元,认为这是一个明智的策略。[@nearcyan](https://twitter.com/nearcyan/status/1783262638778278240)开玩笑说,Meta36亿美元的收入如今都被用于GPU了。
*   **Apple的AI公告**: [@fchollet](https://twitter.com/fchollet/status/1783544742565015954)在Kaggle平台分享了一个针对苹果公司自动评分论文比赛的Keras入门notebook。[@\_akhaliq](https://twitter.com/_akhaliq/status/1783557863069139270)报道了苹果公司发布的CatLIP论文, 重点介绍了利用网络规模的图像-文本数据进行更快预训练的CLIP级视觉识别技术。

* * *

