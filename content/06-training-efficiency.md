# 6. 训练效率

> [!结论] 本章结论
> 视频训练的主要约束是**单样本长序列**（数万到十万 token），所以标准配方是 **FSDP/HSDP 切参数 + 序列 / 上下文并行切序列 + 选择性重计算 + 离线 latent + 分桶均衡**。最省钱的一步往往是课程本身：Open-Sora 2.0 把大部分迭代放在 256px。

## 6.1 并行策略

| 维度 | 方案 | 说明 |
|---|---|---|
| 参数 / 优化器状态 | FSDP2 / ZeRO-3、HSDP | 节点内全切、节点间复制，减少跨机通信 |
| 序列 | DeepSpeed-Ulysses（all-to-all 切 head）、Ring Attention（切序列块）、USP（二者组合） | Wan 2.1 用 2D context parallel；Ulysses 并行度受 head 数限制，超出部分交给 Ring |
| 张量 | Megatron TP | 大模型可用，但视频 DiT 中常被 CP + FSDP 取代 |
| 专家 | EP | Wan 2.2 的去噪阶段 MoE 不需要 EP（两个专家按时间步切换）；token 级 MoE 才需要 |
| 注意力内核 | FlashAttention-3、FlexAttention、MagiAttention（MAGI-1） | 变长 + 块因果 mask 场景需要灵活 mask 的内核 |

## 6.2 显存与算力

- **激活重计算**：选择性 checkpoint（只存注意力输出等贵的激活），配合 CPU offload；Seedance 1.0 使用多级激活 checkpoint。
- **低精度**：BF16 为基线；FP8 GEMM（Transformer Engine）在 H100 / H200 / B200 上可提速 1.3–1.5×，注意归一化和调制层保留高精度。
- **优化器**：Muon 按 HunyuanVideo 1.5 报告可省一半步数；分布式 Muon 需在 FSDP 切片上做正交化，实现有工程成本。
- **算子融合**：RMSNorm、RoPE、AdaLN 调制用 Triton / CUDA 融合，是常见的 5–10% 收益来源。

## 6.3 数据侧效率

- **离线编码**：提前把视频编成 VAE latent、文本编成 embedding（Seedance 1.0 离线完成），训练时只读张量；代价是存储和数据增强受限。
- **分桶 + 打包**：按帧数 × 分辨率 × 宽高比分桶，每桶自动搜 batch size（Open-Sora 2.0）；或等 token 长度序列打包（Waver SPFHP、NaViT 思路），减少 padding 与卡间负载不均。
- **运行时负载均衡**：不同卡拿到不同长度样本会等最慢的那张；Seedance 1.0 采用运行时感知的均衡调度。
- **容错**：万卡规模下需要快速异步 checkpoint 和故障节点自动摘除。

## 6.4 成本参考

| 模型 | 规模 | 训练算力 |
|---|---|---|
| Open-Sora 2.0 | 11B | 约 4160 H200 GPU-天，约 20 万美元 |
| Koala-36M 实验模型 | — | 256×256、2s 片段，看过 1.4 亿样本 |

> [!注意] 闭源模型
> 头部闭源模型未公布算力，普遍估计在千卡到万卡、数月量级（未经核实）。
