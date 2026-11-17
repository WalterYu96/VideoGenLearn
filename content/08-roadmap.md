# 8. 学习路线与参考资料

> [!结论] 建议
> 用 **4 周**补齐：先读开源报告建立全景，再按自己的强项（数据 + RL）切入。

## 4 周学习路线

| 周 | 主题 | 阅读 | 动手 |
|---|---|---|---|
| 第 1 周 | 全景与架构 | 精读 Wan 2.1 与 HunyuanVideo 1.5 报告 | 跑通 Wan 2.2 TI2V-5B 推理和 LoRA 微调 |
| 第 2 周 | 数据 | Seedance 1.0 / Waver / Open-Sora 2.0 / Koala-36M 数据章节 | 搭一个小规模切镜 + 打分 + caption 管线（参考 [镜头切分](#/data-shot)、[去重](#/data-dedup)） |
| 第 3 周 | 后训练 | DanceGRPO / MixGRPO / VideoAlign / 后训练框架论文 | 在 Wan 1.3B 上复现一次 GRPO |
| 第 4 周 | 蒸馏与推理 | rCM、Self-Forcing、TurboDiffusion；再读 Seedance 1.5 pro / 2.0、LTX-2、Kling-Omni | 了解音画与多参考 |

## 参考资料

### 技术报告

- [Wan: Open and Advanced Large-Scale Video Generative Models（Wan 2.1）](https://arxiv.org/abs/2503.20314)
- [HunyuanVideo 1.5 Technical Report](https://arxiv.org/html/2511.18870v2)
- [Seedance 1.0](https://arxiv.org/html/2506.09113)
- [Seedance 1.5 pro](https://arxiv.org/abs/2512.13507v2)
- [Seedance 2.0: Advancing Video Generation for World Complexity](https://www.emergentmind.com/papers/2604.14148)
- [Seedance 2.0 官方发布](https://seed.bytedance.com/en/blog/seedance-2-0-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83)
- [Kling-Omni Technical Report](https://arxiv.org/html/2512.16776v1)
- [LTX-2: Efficient Joint Audio-Visual Foundation Model](https://www.emergentmind.com/papers/2601.03233)
- [Open-Sora 2.0](https://arxiv.org/html/2503.09642)
- [Waver](https://arxiv.org/html/2508.15761)
- [Koala-36M](https://arxiv.org/html/2410.08260)

### 后训练与加速

- [A Systematic Post-Train Framework for Video Generation](https://www.alphaxiv.org/abs/2604.25427.md)
- [VGGRPO: 4D Latent Reward](https://arxiv.org/html/2603.26599)
- [OPSD-V: On-Policy Self-Distillation](https://huggingface.co/papers/2607.08766)
- [TurboDiffusion](https://arxiv.org/html/2512.16093v1)

### 架构细节与 OPD

- [Wan 2.1 GitHub](https://github.com/Wan-Video/Wan2.1) · [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Step-Video-T2V](https://arxiv.org/html/2502.10248)
- [MAGI-1](https://arxiv.org/html/2505.13211)
- [HunyuanVideo（13B）](https://arxiv.org/html/2412.03603)
- [DiffusionOPD](https://arxiv.org/html/2605.15055v1)
- [On-Policy Adversarial Flow Distillation](https://arxiv.org/html/2605.26105v1)
- [MILD: On-Policy Distillation from Image to Video](https://arxiv.org/abs/2609.34371)
- [Rethinking On-Policy Distillation（LLM）](https://arxiv.org/pdf/2604.13016)

### 数据专题补充（本站新增）

- [Kling-MotionControl Technical Report](https://arxiv.org/html/2603.03160v1)
- [EmbeddingGemma 2 开发者指南](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/)
- [EmbeddingGemma 2 多模态 Embedding 文档](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)

### 模型格局（第三方，仅作参考）

- [VideoGen: Best AI Video Models in 2026](https://videogen.io/best-ai-video-models)
- [Wan 3.0: what is confirmed](https://aivideosensei.com/guides/wan-3-0-what-is-confirmed)
- [Is Wan 3.0 open source](https://wavespeed.ai/blog/video-model-access/is-wan-3-0-open-source/)
- [Turing Post: open-source video models](https://www.turingpost.com/p/6-opensource-video-generation-models)
- [Thunder Compute: open-source video models](https://www.thundercompute.com/blog/best-open-source-ai-video-generation-models)

> [!注意] 引用前核对
> 文中提到但未单独链接的方法（Flow-GRPO、DanceGRPO、MixGRPO、DiffusionNFT、DMD2、Self-Forcing、CausVid、Phantom、VACE、OmniHuman、CameraCtrl 等）均为公开 arXiv 论文，描述基于截至 2026 年中的公开资料，建议引用前核对原文。
