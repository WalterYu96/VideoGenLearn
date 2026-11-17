# 7. 推理效率

> [!结论] 本章结论
> 推理加速由四个可以叠乘的方向组成：**少步蒸馏（十倍级）、稀疏注意力（2–5×）、量化（1.5–2×）、并行与缓存**。TurboDiffusion 在单张 RTX 5090 上把 Wan 2.1-T2V-14B-720P 从 4767s 压到 24s（约 200×），就是四者叠加的结果。

```flow
少步采样 | 50–150 NFE → 4–10 NFE
CFG 蒸馏 | 前向次数减半
稀疏注意力 | 静态窗口 / 动态 Top-K
量化 | 低比特注意力、W8A8、FP8
缓存与并行 | 特征缓存、参考 KV 缓存、xDiT / USP
级联超分与 VAE 解码 | 低分辨率主模型 + 少步 VSR；瘦身解码器
```

## 7.1 加速方法分类

| 方向 | 代表方法 | 说明 |
|---|---|---|
| 少步采样 | rCM、DMD2、TSCD、LightX2V 4 步 LoRA | 见 [4.4 蒸馏](#/posttraining)；工业端 50–150 NFE → 4–10 NFE |
| CFG 蒸馏 | guidance embedding | 前向次数减半 |
| 静态稀疏注意力 | STA / SSTA（滑动块）、Radial Attention、窗口注意力 | HunyuanVideo 1.5 SSTA：720p 241 帧每步 5.51s → 2.95s，端到端 1.87× |
| 动态 / 可训练稀疏 | Sparse VideoGen、VSA、SLA（稀疏 + 线性注意力） | SLA 在 TurboDiffusion 里用 Top-K 0.1，即 90% 稀疏 |
| 量化 | SageAttention 2 / 2++ / 3（低比特注意力）、W8A8 INT8、FP8 GEMM | Kling-Omni 用 FP8 GEMM + FP8 注意力 + FP8 通信 |
| 特征缓存 | TeaCache、FasterCache、PAB、First-Block Cache | 相邻步输出变化小时复用；与少步蒸馏兼容性差 |
| 参考 KV 缓存 | 参考 token 跨步复用 | Kling-Omni 约 2× |
| 并行推理 | xDiT / USP、Ulysses + TP 混合、通信计算重叠 | 降低单条视频延迟 |
| 级联超分 | 低分辨率主模型 + 少步 latent VSR | HunyuanVideo 1.5、Seedance、Kling、LTX-2 都采用 |
| VAE 解码 | 瘦身解码器、tiling、并行解码 | Seedance 1.0 解码约 2× |
| 自回归流式 | Self-Forcing、CausVid | 首帧延迟亚秒级，适合实时交互 / 世界模型 |

## 7.2 实测参考

| 场景 | 加速前 | 加速后 | 来源 |
|---|---|---|---|
| Wan 2.1-T2V-14B 720P，RTX 5090 | 4767s | 24s | [TurboDiffusion](https://arxiv.org/html/2512.16093v1) |
| Wan 2.2-I2V-A14B 720P，RTX 5090 | 4549s | 38s | TurboDiffusion |
| Wan 2.1-T2V-1.3B 480P，RTX 5090 | 184s | 1.9s | TurboDiffusion（FastVideo 为 5.3s） |
| Seedance 1.0，5s 1080p，L20 | — | 41.4s | [Seedance 1.0](https://arxiv.org/html/2506.09113) |
| LTX-2 vs Wan 2.2-14B，720p 121 帧 | 22.3 s/步 | 1.22 s/步 | [LTX-2](https://www.emergentmind.com/papers/2601.03233)（架构差异导致，非同模型加速） |
| HunyuanVideo 1.5，720p 121 帧 | — | 峰值显存 13.6GB | [HunyuanVideo 1.5](https://arxiv.org/html/2511.18870v2) |

> [!注意] 读数注意
> TurboDiffusion 的数字不含文本编码和 VAE 解码，且质量只有样例帧、无定量指标。

## 7.3 开源推理框架

- **LightX2V**：面向 Wan / Hunyuan 的量化 + 蒸馏 + 并行一体化推理。
- **FastVideo**（Hao AI Lab）：稀疏注意力（STA / VSA）+ 蒸馏的训练与推理。
- **xDiT**：USP / PipeFusion 等 DiT 并行推理。
- **TurboDiffusion**：SageSLA + rCM + W8A8 一体化（[GitHub](https://github.com/thu-ml/TurboDiffusion)）。
- **diffusers / DiffSynth-Studio / ComfyUI**：生态与研究验证。
