# 2. 预训练：模型架构

> [!结论] 本章结论
> 主流配方是 **因果 3D VAE（4×16×16 成为新默认）+ 全 3D 注意力 DiT / MMDiT + LLM/VLM 文本编码器 + Flow Matching + 图 → 低分辨率视频 → 高分辨率的课程式训练**。2026 年的增量是：音频流分支、MoE 专家、自回归（块因果）形态。

## 2.1 VAE / 视频 tokenizer

**要点**

- **压缩率 vs 可生成性**：压得越狠越快，但 latent 更难建模。近期做法是加语义正则（对齐 DINOv2 / VLM 特征、VA-VAE 思路），让 latent 更“好学”。
- **因果性**：首帧单独编码 → 图片和视频共用一个 VAE，这也是 I2V 和长视频分块的前提。
- **训练三步法**（Wan 2.1）：2D 图像 VAE 膨胀成 3D → 低分辨率视频 → 加 GAN loss 微调。
- **解码是推理瓶颈之一**：Seedance 1.0 用瘦身解码器提速约 2×；常用时空 tiling。

| 模型 | 压缩（T×H×W） | latent 通道 | 备注 |
|---|---|---|---|
| Wan 2.1 VAE | 4×8×8 | 16 | 127M 参数，因果，首帧只做空间压缩；RMSNorm 替代 GroupNorm，支持无限长分块编码 |
| Wan 2.2 VAE | 4×16×16 | 48 | 用于 TI2V-5B，配合 patch 1×2×2 实现总压缩 4×32×32 |
| Seedance 1.0 | 4×16×16 | 48 | MAGVIT 风格，DiT 侧不再 patchify |
| HunyuanVideo 1.5 | 4×16×16 | 32 | 因果 3D VAE |
| Open-Sora 2.0 Video DC-AE | 4×32×32 | 128 | 深压缩，patch=1；速度快但重建细节下降 |
| LTX-Video | 8×32×32 | 128 | 1:192 压缩，解码器内做最后一步去噪 |

### VAE 训练与设计细节

| 模型 | 结构 | 损失与训练 |
|---|---|---|
| HunyuanVideo（13B 版） | 因果 3D 卷积，4×8×8，16 通道；从零训练，不用图像 VAE 初始化 | L1 + 0.1·LPIPS + 0.05·GAN + 1e-6·KL；视频 : 图像 = 4 : 1 |
| MAGI-1 | Transformer VAE：8×8×4 的 3D 卷积嵌入到 1024 维，接 24 个带 QK-norm 的 Transformer block，输出 32 通道（16 均值 + 16 对数方差）；解码端用 pixel shuffle | 阶段一：16 帧 256²；阶段二：图像与 16 帧视频混训，总像素约 256² 或 384²，宽高比在 [0.25, 4] 均匀采样；L1 + KL + LPIPS + GAN；推理用 256 窗口、步长 192 的空间 tiling |
| Step-Video-T2V | 双路径融合，因果 3D 卷积，16×16×8 | — |
| Seedance 1.0 | MAGVIT 风格因果卷积，4×16×16，48 通道；借鉴 DC-AE 去掉 DiT 侧 patchify | L1 + KL + LPIPS + 对抗损失，PatchGAN 式混合判别器 |
| Wan 2.1 | 因果 3D 卷积，4×8×8，16 通道，127M 参数 | 2D→3D 膨胀 → 低分辨率视频 → GAN 微调；可分块编解码任意长度 1080P |

### 选型时先算 token 数

$$N_{tok} = \Big(\frac{T}{r_t}+1\Big)\cdot\frac{H}{r_s\,p}\cdot\frac{W}{r_s\,p}$$

- 720p × 121 帧：4×8×8 VAE 加 2×2 patch → 31×45×80 ≈ **11.2 万 token**；4×16×16 VAE 不 patchify 也是 11.2 万。两者 token 数相同，区别在于压缩发生在 VAE 还是 patch embedding。
- 4×16×16 VAE 再加 2×2 patch（Wan 2.2 TI2V-5B）→ 31×23×40 ≈ **2.9 万 token**，注意力成本约降为 1/16。
- 通道越多，重建越好，但扩散越难学。高压缩 VAE 通常要配更大的 DiT 或语义对齐正则。

## 2.2 主干架构范式

| 范式 | 代表 | 特点 |
|---|---|---|
| DiT + 文本交叉注意力 | Wan 2.1 / 2.2 | 视频 token 自注意力 + cross-attn 注入文本；结构简单，文本长度不影响序列长度 |
| MMDiT（双流 → 单流） | HunyuanVideo、Open-Sora 2.0、Waver 12B（16 双流 + 40 单流） | 文本与视觉 token 拼接做联合注意力，前若干层分开权重；与 FLUX、Z-Image 思路一致 |
| 时空解耦 | Seedance 1.0 | 空间层做 MMDiT 联合注意力，时间层做窗口注意力，省算力 |
| 去噪阶段 MoE | Wan 2.2 A14B | 高噪专家管布局、低噪专家管细节，按 SNR 切换；总 27B、每步激活 14B |
| 音视频双流 | LTX-2（14B 视频 + 5B 音频）、Seedance 1.5 pro、Ovi | 两条模态流双向交叉注意力，时间 RoPE 对齐，共享 timestep 条件 |
| 统一多任务 | Kling-Omni、Waver、Seedance 2.0 | 噪声 latent + 条件帧 + mask 拼接，一个模型支持 T2V / I2V / 参考 / 编辑 |
| 自回归 / 块因果扩散 | MAGI-1、Self-Forcing、CausVid、LongLive | 按 chunk 从左到右生成，可用 KV cache，适合实时与长视频 |

**位置编码**：3D RoPE（T/H/W 分轴分配频率）是标配；Waver 叠加可学习分解位置编码；多参考输入时常给参考图分配独立的时间索引或类型 embedding。

**文本编码器演进**：T5-XXL / umT5（Wan）→ T5 + LLM 双编码器（Waver：flan-T5-XXL + Qwen2.5-32B）→ VLM（HunyuanVideo 1.5：Qwen2.5-VL + Glyph-ByT5 专管文字渲染）→ 大 LLM 多层特征聚合（LTX-2：Gemma3-12B 全层聚合 + 可学习 thinking tokens）。VLM 编码器的好处是能直接吃参考图。

## 2.3 代表模型结构超参

| 模型 | 参数 | 层数 | 隐藏维 | FFN 维 | 头数 × 头维 | 主干结构 | 文本编码 / 注入 |
|---|---|---|---|---|---|---|---|
| Wan 2.1-1.3B | 1.3B | 30 | 1536 | 8960 | 12 × 128 | 全 3D 自注意力 | umT5-XXL，交叉注意力 |
| Wan 2.1-14B | 14B | 40 | 5120 | 13824 | 40 × 128 | 全 3D 自注意力 | umT5-XXL，交叉注意力 |
| Wan 2.2-A14B | 27B 总 / 14B 激活 | 两个 14B 专家 | — | — | — | 去噪阶段 MoE | umT5-XXL |
| HunyuanVideo 1.5 | 8.3B | 54 | 2048 | 8192 | 16 × 128 | 双流 block | Qwen2.5-VL + Glyph-ByT5 |
| Step-Video-T2V | 30B | 48 | 6144 | 24576 | 48 × 128 | 全 3D 自注意力 + 交叉注意力 | Hunyuan-CLIP + Step-LLM，序列拼接后交叉注意力 |
| MAGI-1-4.5B | 4.5B | 34 | 3072 | 12288 | GQA | 块因果自回归 | T5，交叉注意力 |
| MAGI-1-24B | 24B | 48 | 6144 | 16384 | GQA，8 个 KV 组 | 块因果自回归，SwiGLU | T5，交叉注意力 |
| Waver | 12B | 16 双流 + 40 单流 | — | — | 24 × 128 | MMDiT | flan-T5-XXL + Qwen2.5-32B |
| Open-Sora 2.0 | 11B | 19 双流 + 38 单流 | 3072 | — | 24 × 128 | MMDiT（FLUX 结构） | T5-XXL + CLIP-L |
| LTX-2 | 14B 视频 + 5B 音频 | — | — | — | — | 非对称双流 | Gemma3-12B |

> [!来源] 数字出处
> Wan 2.1 的数字来自 [官方 GitHub](https://github.com/Wan-Video/Wan2.1)，Step-Video 来自 [论文](https://arxiv.org/html/2502.10248) Table 1。MAGI-1 和 Open-Sora 2.0 的部分数字取自论文页摘要，未逐字核对。头维 128 是共识。

## 2.4 DiT Block 内部结构

**典型数据流（Wan 式）**

```flow
x → AdaLN(shift, scale)
3D 自注意力 | QK-norm + 3D RoPE
gate 残差
文本交叉注意力
AdaLN → FFN
gate 残差
```

- **timestep 调制**：Wan 只用一个跨 block 共享的 MLP（Linear + SiLU）输出 6 个调制参数（两组 shift / scale / gate），每个 block 只学一份偏置，比原版 DiT 每层独立 AdaLN 省很多参数。Step-Video 用 AdaLN-Single。MAGI-1 去掉输入调制，并把调制值 Softcap 到 [−1, 1]，以稳定大模型训练。
- **归一化**：QK-norm 几乎是标配，防止注意力 logit 爆炸；RMSNorm 取代 LayerNorm；MAGI-1 在 FFN 外再加 sandwich norm。
- **FFN**：GELU（Wan、Step-Video）或 SwiGLU（MAGI-1 24B）。
- **注意力**：MAGI-1 用 GQA 降低 KV 开销，自注意力与交叉注意力共享 Q 投影。

**文本注入的三种方式**

1. **交叉注意力**：Wan、Step-Video、MAGI-1。文本长度不进入自注意力序列，计算省。
2. **联合注意力（MMDiT）**：HunyuanVideo、Waver、Open-Sora 2.0。图文交互更强，与 FLUX、Z-Image 同源。
3. **时空解耦下只在空间层拼文本**：Seedance 1.0 的空间层做帧内 MMDiT 联合注意力（视觉与文本 token 用独立的 AdaLN、QKV、MLP 权重），时间层只处理视觉 token，并在帧内做窗口划分。

**位置编码细节**：3D RoPE 把 head 维按通道切给 T/H/W 三轴，各做 1D RoPE 后拼接（Step-Video）。MAGI-1 的 base 频率可学习。Seedance 的 MM-RoPE 给视觉 token 用 3D RoPE、给文本 token 再加 1D 位置，以支持多镜头交错序列，各镜头按时间排序、各配一段 caption。

## 2.5 条件注入与特殊结构

- **I2V 注入**
  - **通道拼接**：noisy latent ⊕ 条件帧 latent（无条件位置补零）⊕ 二值 mask，统一 T2I / T2V / I2V（Seedance 1.0、Waver）。
  - **通道拼接 + 语义特征**：HunyuanVideo 1.5 再把 SigLIP 特征拼进序列，并加可学习类型 embedding。
  - **续写式**：MAGI-1 没有专门的 I2V 模块，I2V 就是“首个 chunk 第一帧干净”的视频续写；干净 chunk 加至多 5% 的噪声，并去掉文本条件。
- **去噪阶段 MoE**（Wan 2.2）：高噪专家负责早期去噪（布局、运动），低噪专家负责后期细节，按 SNR 阈值 $t_{moe}$ 切换（官方 README：阈值取 SNR_min 一半所对应的步）。每步只激活一个专家，计算量约等于 14B，但两个专家都要常驻显存或按需换入。
- **块因果自回归**（MAGI-1）：每 24 帧一个 chunk，chunk 内全注意力、chunk 间因果。每个 chunk 有独立 timestep，越早的 chunk 噪声越低（$i<j$ 时 $t_i<t_j$）。推理时最多 4 个 chunk 同时流水去噪，并发出前一段成片。
- **级联 refiner**：Seedance 1.0 基模出 480p，refiner 以上采样后的低分辨率 latent 为条件生成 720p / 1080p；Waver 的 refiner 用窗口注意力，只有首尾各 8 层是全注意力。
- **音频双流**（LTX-2）：跨模态 AdaLN 共享 timestep，双向交叉注意力里加时间 RoPE 对齐音画；推理时文本与跨模态各用一个 CFG scale。

## 2.6 训练目标：Flow Matching

线性插值路径 + 速度预测（rectified flow）：

$$x_t = (1-t)\,x_0 + t\,\epsilon,\qquad \mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\big\|v_\theta(x_t,t,c) - (\epsilon - x_0)\big\|^2$$

timestep shift（SD3 式，$s$ 越大越偏向高噪声）：

$$t' = \frac{s\,t}{1+(s-1)\,t}$$

- **timestep 采样**：logit-normal 或分辨率相关 shift（分辨率和帧数越大，shift 越大，偏向高噪声），与 SD3 / FLUX 一致。Seedance 1.0 先 logit-normal 采样，再按分辨率和时长做 shift。MAGI-1 用 logit-normal（$m=0$，$s=0.5$）加 shift，约 70% 训练算力落在论文记法下 $t<0.3$ 的区间。
- **推理 shift**：Wan 2.1 在 diffusers 中 720P 用 5.0、480P 用 3.0，官方脚本 T2V-1.3B 建议 8–12；采样 50 步（T2V-14B）或 40 步（I2V-14B），CFG 5–6。
- **蒸馏阶段**：Step-Video 改用 U 形 timestep 分布（$a=5$），加重两端。
- **自回归模型**：MAGI-1 每个 chunk 独立采样 timestep，并约束前面的 chunk 更干净。
- **表征对齐（REPA 类）**：Waver 在 480p 阶段把 DiT 第 16 层特征与 Qwen2.5-VL 特征做余弦对齐（权重 0.5），加快收敛、提升语义。
- **优化器**：AdamW 为主（Wan 2.1：lr 1e-4，wd 1e-3）；HunyuanVideo 1.5 改用 Muon，称用一半步数达到更低 loss。
- **条件 dropout**：文本约 10% 置空（常见设置）；Open-Sora 2.0 的图像条件 dropout 是 12.5%，推理时图像 CFG 3、文本 CFG 7.5。

## 2.7 多阶段课程与优化超参

| 阶段 | Wan 2.1 | HunyuanVideo 1.5 | Seedance 1.0 |
|---|---|---|---|
| ① 图像预热 | T2I 256px | T2I 256p → 512p | — |
| ② 低分辨率图视频联合 | 256px | 256p，T2I : T2V : I2V = 1 : 6 : 3 | 256px，3–12s，12fps |
| ③ 中分辨率 | 480px | 480p | 640px |
| ④ 高分辨率 / 高帧率 | 720px | 720p → 720p 24fps | 24fps |
| 超分 | — | 级联 VSR 到 1080p | 级联 refiner 到 1080p |

**Step-Video-T2V 训练配方**（batch 为每节点）

| 阶段 | 分辨率 | 迭代数 | batch | LR | 样本量 |
|---|---|---|---|---|---|
| T2I 预训练 | 256×256 | 25.3 万 | 40 | 1e-4 | 38 亿 |
| T2VI 低分辨率 | 192×320 | 43 万 | 4 | 6e-5 | 6.44 亿 |
| T2VI 高分辨率 | 544×992 | 4.6 万 | 2 → 1 | 2e-5 → 1e-5 | 2730 万 |
| SFT → DPO | — | — | — | — | — |

帧数分桶为 1 / 68 / 136 / 204 帧（1 帧即图像）。并行用 TP8 + 序列并行 + ZeRO-1，MFU 32%，在数千张 H800 上训练。

**MAGI-1 的分辨率课程**：4.5B 依次训练 360p（≤ 8s）、480p（≤ 8s）、720p（≤ 16s），全程图像视频联合训练；24B 从 256p 起步。

**优化器超参参考**

| 模型 | 优化器 | 峰值 LR | betas | weight decay | warmup |
|---|---|---|---|---|---|
| Wan 2.1 | AdamW | 1e-4 | — | 1e-3 | — |
| MAGI-1 | AdamW | 1e-4（4.5B 第三阶段降至 1e-5） | 0.9 / 0.95 | 0.1 | 4.5B：1000 步；24B：1 万步 |
| Step-Video-T2V | — | 1e-4 → 6e-5 → 2e-5 | — | — | — |
| HunyuanVideo 1.5 | Muon | — | — | 0.01 | — |

> [!经验] 规律
> - LR 随分辨率阶段递减；batch 按 token 数而非样本数对齐，高分辨率阶段每卡样本数降到 1–2；模型越大 warmup 越长。
> - 图像数据全程混训（保持画质和概念覆盖）；I2V 比例逐步上调（Seedance 1.0 预训练 20%、继续训练 40%）；1080p 以上常交给级联超分而非主模型。
> - 成本参考：Open-Sora 2.0 用约 4160 GPU-天（H200，约 20 万美元）训出 11B 模型，其中 256px T2V 阶段就占 2240 GPU-天。
