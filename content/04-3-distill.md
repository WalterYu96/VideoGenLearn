# 4.3 蒸馏：步数、CFG 与自回归蒸馏

> [!结论] 本节结论
> 一个开源视频底模默认要 **50 步 × 2（CFG）= 100 NFE**，蒸馏要同时压三件事：**CFG 的第二次前向、去噪步数、双向注意力（改成块因果才能流式）**。2025–2026 的主流配方是：**一致性 / 轨迹蒸馏打底 → 分布匹配（DMD 系）或对抗提质 → 需要实时就再做自回归蒸馏（Self-Forcing 系）**。工业端普遍做到 **4–10 NFE、端到端 10× 以上**（Seedance 1.0 / 1.5 pro、Kling-Omni 150→10 NFE），开源 Wan 上 4 步学生的 VBench 已经能持平甚至略超教师（rCM、CoDMD）。代价集中在**多样性和运动幅度下降**，这是 2026 年蒸馏研究的主战场。

```flow
教师 | 50 步 × CFG = 100 NFE，双向注意力
CFG 蒸馏 | guidance 作为条件输入，去掉无条件分支，约 2×
步数蒸馏 | 一致性 / 轨迹（PCM、TSCD、rCM）→ 4–10 步
提质 | 分布匹配（DMD2、TDM）或对抗（APT、LADD），保留轨迹损失作正则
可选：自回归蒸馏 | 块因果 ODE 初始化 → Self-Forcing 视频级 DMD → KV cache 流式
系统叠加 | 稀疏注意力、量化、并行，见第 7 章
```

> [!提示] 本节与其他章节的分工
> 推理侧的稀疏注意力、量化、缓存和各框架实测速度见 [第 7 章 推理效率](#/inference)；DMD / Self-Forcing 作为“学生自采样 + 教师稠密监督”的在线策略蒸馏视角，见 [4.4 OPD](#/post-opd)；蒸馏前的 RL 见 [4.2 RL](#/post-rl)。

## 先算账：NFE 从哪里来

| 来源 | 典型开销 | 用什么去掉 | 收益 |
|---|---|---|---|
| CFG | 每步条件 + 无条件两次前向 | CFG 蒸馏（guidance embedding），或在步数蒸馏时一并蒸掉 | 约 2×；HunyuanVideo（v1）报告文本 guidance 蒸馏约 1.9×（[来源](https://arxiv.org/abs/2412.03603)） |
| 去噪步数 | 40–50 步（Wan 2.1 教师记为 50×2 NFE，[来源](https://arxiv.org/html/2510.08431)） | 步数蒸馏 | 1–4 步时 15–50×（rCM，[来源](https://arxiv.org/abs/2510.08431)） |
| 双向注意力 | 整段视频一起去噪，首帧要等全部生成完 | 自回归蒸馏（块因果 + KV cache） | 首帧延迟降到亚秒级，单卡 16–23 FPS 流式（见 4.3.4） |

## CFG 蒸馏

CFG 每步要算两次：

$$\hat v = v_\theta(x_t,\varnothing) + w\,\big(v_\theta(x_t,c) - v_\theta(x_t,\varnothing)\big)$$

CFG 蒸馏训练一个把 $w$ 当条件输入的学生，一次前向就拟合 $\hat v$：

$$\mathcal L = \mathbb E_{t,w}\big\lVert v_\phi(x_t, c, w) - \hat v \big\rVert^2$$

| 做法 | 要点 | 例子 |
|---|---|---|
| 两阶段：先蒸 guidance 再蒸步数 | 先让单模型匹配 CFG 输出，再渐进蒸馏步数；像素空间 4 步、最高 256× 提速，latent 模型 1–4 步、至少 10× | Meng et al.（[来源](https://arxiv.org/abs/2210.03142)） |
| guidance embedding（发布“guidance-distilled”权重） | $w$ 像时间步一样做 embedding 注入；推理时 guidance 只是一个输入值 | FLUX.1 [dev]（[来源](https://huggingface.co/black-forest-labs/FLUX.1-dev)）；HunyuanVideo 1.5 的测速全部用 CFG 蒸馏模型（[来源](https://arxiv.org/html/2511.18870v2)），官方单独发布 cfg-distill 权重（[来源](https://huggingface.co/tencent/HunyuanVideo-1.5)） |
| 与步数蒸馏合并 | 教师用 CFG 采样出目标，学生直接学带 guidance 的结果，推理设 CFG=1 | rCM（guidance 被“同时蒸进学生”，[来源](https://arxiv.org/html/2510.08431)）、LightX2V 4 步模型（[来源](https://lightx2v-en.readthedocs.io/en/latest/_sources/method_tutorials/step_distill.md)）、LTX-2 distilled（8 步、CFG=1，[来源](https://huggingface.co/Lightricks/LTX-2)） |
| 多步蒸馏里只在最细步长加 CFG 蒸馏 | 推理最后阶段改 guidance 系数以减轻过饱和 | MAGI-1（[来源](https://arxiv.org/html/2505.13211)） |

> [!注意] CFG 蒸馏的副作用
> 蒸掉 CFG 后负向提示词基本失效，guidance 强度也只能在训练覆盖的范围内调。社区在 CFG 蒸馏 / 步数蒸馏模型上普遍要求 guidance_scale=1，再开 CFG 容易偏色、过饱和（[来源](https://huggingface.co/lightx2v/Hy1.5-Distill-Models)）。音视频模型的 CFG 更复杂：LTX-2 底模对视频和音频用不同的文本 guidance（3 / 7）和跨模态 guidance（[来源](https://arxiv.org/html/2601.03233)），蒸馏时这些都要被学生吸收。

## 步数蒸馏的四大类

| 类别 | 学什么 | 散度 / 性质 | 优点 | 缺点 |
|---|---|---|---|---|
| 轨迹 / 一致性 | 教师 PF-ODE 上任意点 → 终点（或分段终点） | 偏 mode-covering（前向） | 稳定、保多样性、可离线 | 少步时细节糊，误差累积 |
| 分布匹配 | 学生样本分布 ≈ 教师分布 | 反向 KL，mode-seeking | 1–4 步画质高，可超过教师 | 多样性 / 运动下降，需要 fake score 网络 |
| 对抗 | 判别器区分学生样本与真实 / 教师样本 | GAN | 锐度、细节好；可接人类偏好数据 | 训练不稳，调参多 |
| 从头训练少步 | 平均速度 / 步长条件的流 | 无需教师 | 一个模型多种步数 | 大模型上多作为蒸馏目标使用 |

### 轨迹 / 一致性类

一致性模型要求同一条 ODE 轨迹上的点映射到同一终点：$f_\theta(x_t,t) = f_\theta(x_{t'},t')$。分段版本（PCM、TSCD）只要求映射到所在分段的终点，更容易学。

| 方法 | 核心 | 关键数字 / 用途 |
|---|---|---|
| LCM | latent 空间一致性蒸馏，CFG 一起蒸 | 图像 2–4 步（[来源](https://arxiv.org/abs/2310.04378)） |
| PCM | 轨迹分段，每段内做一致性 | Kling-Omni 第一阶段的参考方法之一（[来源](https://arxiv.org/abs/2405.18407)） |
| Hyper-SD（TSCD） | 分段一致性，段数逐步减少，再加人类反馈和 score 蒸馏 | Seedance 1.0 用 TSCD 得到约 4× 加速且效果持平（[来源](https://arxiv.org/html/2506.09113v2)，[Hyper-SD](https://arxiv.org/abs/2404.13686)） |
| sCM（OpenAI） | 连续时间一致性，TrigFlow 参数化 | 2 步 ImageNet 512 FID 1.88，1.5B（[来源](https://arxiv.org/abs/2410.11081)） |
| rCM（NVIDIA，ICLR 2026） | sCM + score 蒸馏作“长跳”正则（补一个反向散度项）；自写 FlashAttention-2 JVP kernel，扩到 10B+ | Wan 2.1 14B / Cosmos，1–4 步，15–50×；质量与 DMD2 持平、多样性更好（[来源](https://arxiv.org/abs/2510.08431)，[代码](https://github.com/NVlabs/rcm)） |
| Shortcut | 网络额外以步长 $s$ 为条件，“一大步 = 两小步”自举 | MAGI-1 用它蒸馏：步长从 {1/64, 1/32, 1/16, 1/8} 循环采样，同一模型可跑 64 / 32 / 16 / 8 步（[来源](https://arxiv.org/html/2505.13211)，[Shortcut](https://arxiv.org/abs/2410.12557)） |
| MeanFlow | 学区间平均速度 $u(z_t,r,t)=\frac{1}{t-r}\int_r^t v\,d\tau$，用恒等式 $u = v-(t-r)\tfrac{d}{dt}u$ 训练 | 从头训练 1-NFE ImageNet 256 FID 3.43（[来源](https://arxiv.org/abs/2505.13447)）；Seedance 1.5 pro 的蒸馏框架引用了它（[来源](https://www.alphaxiv.org/abs/2512.13507.md)） |

### 分布匹配类

DMD 只看学生生成的样本：把它加噪后，比较教师（real score）和一个在线拟合学生分布的 fake score 网络的输出差，作为梯度：

$$\nabla_\theta \mathcal L_{DMD} = \mathbb E_{z,t}\Big[\big(s_{\text{fake}}(x_t,t) - s_{\text{real}}(x_t,t)\big)\,\frac{\partial G_\theta(z)}{\partial \theta}\Big]$$

梯度形式与文生 3D 的 VSD（[来源](https://arxiv.org/abs/2305.16213)）同源；SiD（[来源](https://arxiv.org/abs/2404.04057)）是另一条一步分布蒸馏路线。DMD 在学生自己的样本上取教师信号，所以 [4.4 OPD](#/post-opd) 把它看作扩散版 OPD 的前身。

| 方法 | 改进点 | 关键数字 |
|---|---|---|
| DMD | 分布匹配 + 回归损失（需要预生成噪声–图像对） | [来源](https://arxiv.org/abs/2311.18828) |
| DMD2 | 去掉回归损失；两时间尺度更新稳定 fake critic；加 GAN 损失用真实数据；训练时模拟多步推理 | ImageNet-64 一步 FID 1.28，推理成本降 500×（[来源](https://arxiv.org/abs/2405.14867)） |
| TDM | 轨迹分布匹配，学生走少步 ODE（不是 SDE） | PixArt-α 4 步，训练成本为教师的 0.01%；CogVideoX-2B 4 NFE VBench 80.91 → 81.65（[来源](https://arxiv.org/abs/2503.06674)） |
| Phased DMD（商汤，2025-10） | 按 SNR 分段，每段一个专家（少步 MoE），缓解随机梯度截断带来的多样性损失 | 在 Wan 2.2-28B 上提升运动和画质（[来源](https://arxiv.org/abs/2510.27684)） |
| CoDMD（2026-06） | 在 DMD 上加 batch 级 + 帧级相似度矩阵匹配，不加网络 | Wan 2.1 14B 4 步 VBench 84.87（教师 84.12、DMD 83.81、rCM 84.05），710s → 27s；DMD 的 dynamic degree 64.44，CoDMD 81.39（[来源](https://huggingface.co/papers/2606.21982.md)） |
| SMD（2026-10） | 在线策略速度回归，不要 fake score 也不要对抗；扩到 33B，含音视频联合 | 4-NFE 超过多步教师（[来源](https://arxiv.org/abs/2610.18472)） |

### 对抗类

| 方法 | 判别器 | 关键数字 / 用途 |
|---|---|---|
| ADD（SDXL-Turbo） | 冻结 DINOv2 像素判别器 + score 蒸馏 | [来源](https://arxiv.org/abs/2311.17042) |
| LADD（SD3-Turbo） | 用预训练 latent 扩散模型的特征做判别器 | SD3 8B，4 步无 guidance（[来源](https://arxiv.org/abs/2403.12015)） |
| Seaweed-APT（字节） | 蒸馏初始化后直接对真实数据做对抗后训练 | 1 步生成 1280×720、24fps、2 秒视频（[来源](https://arxiv.org/abs/2501.08316)） |
| APT 多步版 + 人类偏好 | 判别器用人类偏好数据训练，压制加速伪影 | Seedance 1.0 蒸馏第三段（[来源](https://arxiv.org/html/2506.09113v2)） |
| DMAD（2026-10） | 把 DMD 改写成对抗：两个判别头替代 fake score 网络，最优判别器下等价于 DMD 梯度 | Wan 2.1-14B 4 步 VBench 85.15；MiniMax-H3-33B 音视频 4 步，人类偏好胜 DMD2 79.1%、胜 rCM 84.6%（[来源](https://arxiv.org/abs/2610.02188)） |

### Wan 2.1 上的同场对比

rCM 论文 Table 2（480p，VBench Total；[来源](https://arxiv.org/html/2510.08431)）：

| 模型 | NFE | Total | Quality | Semantic |
|---|---|---|---|---|
| Wan 2.1 1.3B 教师 | 50×2 | 83.02 | 83.95 | 79.26 |
| DMD2 1.3B | 4 | 84.56 | 85.58 | 80.50 |
| rCM 1.3B | 4 | 84.43 | 85.38 | 80.63 |
| Wan 2.1 14B 教师 | 50×2 | 83.58 | 84.26 | 80.92 |
| rCM 14B | 4 / 2 / 1 | 84.92 / 85.05 / 83.02 | 85.43 / 85.57 / 83.57 | 82.88 / 82.95 / 80.81 |

> [!注意] VBench 分数“超过教师”要打折看
> 少步学生普遍 VBench 不降反升，但 VBench 对多样性、运动幅度不敏感；rCM 自己也承认在 1.3B 上和 DMD2 基本持平，优势主要在定性的多样性对比。CoDMD 用户研究里 CoDMD 对教师的胜率只有 51.3%（[来源](https://huggingface.co/papers/2606.21982.md)）。上线前一定要做人工 GSB 和多样性检查。

## 自回归蒸馏与实时流式生成

目标：把双向教师变成**块因果、少步、带 KV cache** 的学生，边生成边播放。标准三段式（Self-Forcing、Krea Realtime 14B、2026 后训练框架都用这一套）：

```flow
双向少步蒸馏 | 先用 DMD 得到少步双向学生
因果初始化 | 加块因果 mask，用教师 ODE 轨迹做回归（causal ODE）或因果一致性蒸馏
Self-Forcing | 训练时用自己生成的历史做条件，完整 rollout + 视频级 DMD
长视频 | 滚动 KV cache、attention sink、长 rollout 上继续蒸馏
```

关键问题是 **exposure bias**：teacher forcing 训练时看的是真实历史，推理时看的是自己生成的、带误差的历史。Self-Forcing 让训练也在自己的 rollout 上进行，这正是 OPD 的思路（详见 [4.4 OPD](#/post-opd)）。

| 方法 | 时间 | 要点 | 速度 / 长度 |
|---|---|---|---|
| CausVid | 2024-12 | 双向教师 → 因果学生的非对称 DMD；ODE 轨迹初始化；50 步 → 4 步 | 单卡 9.4 FPS，VBench-Long 84.27（[来源](https://arxiv.org/abs/2412.07772)） |
| Self-Forcing | 2025-06 | 训练时自回归 rollout + KV cache，视频级 DMD；随机梯度截断控成本 | 单 H100 17.0 FPS、首帧 0.69s，VBench 84.31（CausVid 81.20）；一次 DMD 约 1.5 小时（64×H100）（[来源](https://arxiv.org/html/2506.08009)） |
| APT2（字节） | 2025-06 | 自回归对抗后训练，每个 latent 帧 1 NFE，student forcing | 8B，单 H100 736×416 / 8×H100 1280×720，24fps 实时，约 1 分钟（[来源](https://arxiv.org/abs/2506.09350)） |
| MAGI-1（Sand AI） | 2025-05 | 原生块自回归预训练（每块 24 帧），shortcut + CFG 蒸馏 | 24B，峰值推理成本不随长度增长（[来源](https://arxiv.org/abs/2505.13211)） |
| LongLive（NVIDIA 等） | 2025-09 | KV-recache 支持中途换 prompt；短窗口注意力 + frame sink；长视频上调 | 1.3B，32 GPU-天调成分钟级；单 H100 20.7 FPS，最长 240s，支持 INT8（[来源](https://arxiv.org/abs/2509.22622)） |
| Rolling Forcing | 2025-09 | 滚动窗口内多帧联合去噪（噪声递增），attention sink | Wan 2.1-1.3B，15.79 FPS、延迟 0.76s；漂移 0.01（Self-Forcing 1.66）（[来源](https://arxiv.org/html/2509.25161)） |
| Self-Forcing++ | 2025-10 | 在学生自己的长 rollout 片段上用短教师监督 | 超过教师时长 20×，最长 4 分 15 秒（[来源](https://arxiv.org/abs/2510.02283)） |
| Krea Realtime 14B | 2025-10 | Self-Forcing 扩到 Wan 2.1 14B；KV cache 每卡可达 25GB，要动态释放 + 梯度检查点 | 单 B200 11 FPS（4 步），首帧约 1s（[来源](https://www.krea.ai/blog/krea-realtime-14b)） |
| Reward Forcing | 2025-12 | EMA-Sink 防止抄首帧；Re-DMD 按 VLM 运动评分加权 | 单 H100 23.1 FPS（[来源](https://arxiv.org/abs/2512.04678)） |
| Causal Forcing++（清华 / 生数） | 2026-05 | 用因果一致性蒸馏初始化，逐帧 1–2 步 | 比 4 步分块版首帧延迟降 50%，第二阶段训练成本约 1/4（[来源](https://arxiv.org/abs/2605.15141)） |
| RAVEN | 2026-05 | 训练时交错干净历史与噪声状态；CM-GRPO 直接在一致性采样步上做 RL | [来源](https://arxiv.org/abs/2605.15190) |
| Causal-rCM（NVIDIA 等） | 2026-06 | teacher forcing = 前向散度（一致性），self forcing = 反向散度（DMD）；结论：teacher-forcing CM 是 self-forcing DMD 最好的初始化 | 2 步因果 Wan 2.1-1.3B VBench 84.63；只用合成数据（[来源](https://arxiv.org/abs/2606.25473)） |

> [!经验] 自回归蒸馏的坑
> 1. **初始化决定上限**：直接上 Self-Forcing 不稳，先做 causal ODE 回归或因果一致性蒸馏（Causal Forcing++、Causal-rCM 都把初始化当成主要瓶颈）。
> 2. **长视频漂移**：短教师只有约 5 秒，超出后靠 attention sink、滚动窗口和长 rollout 蒸馏（Self-Forcing++、LongLive、Rolling Forcing）。
> 3. **抄首帧、运动变小**：sink token 太强会让后续帧复制开头（Reward Forcing 的出发点）。
> 4. **显存**：Self-Forcing 训练要同时放生成器、教师、fake score 等多个模型，14B 规模下 64×H100 也会 OOM（Krea）。

## 各家工业配方

| 模型 | 蒸馏配方 | 步数 / NFE / 加速 | 出处 |
|---|---|---|---|
| Seedance 1.0（字节） | TSCD（约 4×）→ RayFlow score 蒸馏 → 多步 APT（偏好数据判别器）；另有 thin VAE decoder（2×） | 未给 NFE；端到端 10× 以上，L20 上 1080p 5s 视频 41.4s；四个维度与原模型持平 | [报告](https://arxiv.org/html/2506.09113v2)，[RayFlow](https://arxiv.org/abs/2503.07699) |
| Seedance 1.5 pro | 多阶段蒸馏框架，引用 MeanFlow、Hyper-SD、RayFlow；叠加量化和并行 | 未给 NFE；端到端 10× 以上，“无明显质量损失” | [报告](https://www.alphaxiv.org/abs/2512.13507.md) |
| Seedance 2.0 | 提供 Fast 版用于低延迟场景，方法未披露 | 未公开 | [报告](https://www.alphaxiv.org/abs/2604.14148.md) |
| Kling-Omni（快手） | ① 轨迹匹配（PCM / Hyper-SD 式，直接在 10 步调度上训练）② 分布匹配（TDM 式 ODE 学生），保留轨迹损失作正则 | 150 NFE → 10 NFE | [报告](https://arxiv.org/html/2512.16776v1) |
| HunyuanVideo 1.5（腾讯） | CFG 蒸馏；蒸馏阶段同时做稀疏注意力训练；之后单独发布步数蒸馏版 | 50 步 → 8 或 12 步（推荐 12）；官方未给蒸馏加速倍数 | [报告](https://arxiv.org/html/2511.18870v2)，[HF](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| LTX-2（Lightricks） | 发布 distilled 版（音视频联合 19B） | 8 步、CFG=1 | [HF](https://huggingface.co/Lightricks/LTX-2) |
| MAGI-1（Sand AI） | shortcut 蒸馏 + CFG 蒸馏 | 64 → 32 / 16 / 8 步可选 | [报告](https://arxiv.org/html/2505.13211) |
| 2026 系统性后训练框架 | SFT → RLHF → Prompt 增强 → 自回归蒸馏：DMD 少步双向学生 → 块因果 ODE 回归 → Self-Forcing + 视频级 DMD；音视频版沿用 OmniForcing 的 audio sink token | 未给数字 | [报告](https://www.alphaxiv.org/abs/2604.25427.md) |

**Wan 社区生态**（开源底模上最常用的加速方案）

| 方案 | 做法 | 数字 |
|---|---|---|
| LightX2V 4 步模型 / LoRA | 基于 Self-Forcing 代码库改的**双向** DMD，同时蒸 CFG；约 5 万条 prompt；有 Wan 2.1 / 2.2（高噪、低噪两个专家各一个 LoRA）和 HunyuanVideo 1.5 版本 | 40–50 步 → 4 步，约 20–24×（[文档](https://lightx2v-en.readthedocs.io/en/latest/_sources/method_tutorials/step_distill.md)，[Wan2.2 LoRA](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)） |
| CausVid LoRA | 从 CausVid 14B 微调权重里提取的 LoRA，当作步数蒸馏插件用在双向推理里 | 实验性质，无官方指标（[来源](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors)） |
| rCM / TurboDiffusion（清华 + NVIDIA） | rCM 步数蒸馏 + 稀疏线性注意力 + W8A8 | 叠加后约 100–200×，见 [第 7 章](#/inference) |
| FastWan（FastVideo） | “稀疏蒸馏”：VSA 稀疏注意力与 DMD 联合训练，学生稀疏、教师全注意力；训练数据是 Wan 自己生成的合成视频 | Wan 2.1-1.3B 480p 5s 视频单 H200 约 5s（去噪 1s）（[来源](https://haoailab.com/blogs/fastvideo_post_training)） |

## 实践要点

### 顺序：先 RL 后蒸馏，还是在蒸馏模型上做 RL

| 路线 | 依据 | 适用 |
|---|---|---|
| 先 RL 后蒸馏（主流） | 2026 后训练框架把自回归蒸馏放在 RLHF 和 Prompt 增强之后（[来源](https://www.alphaxiv.org/abs/2604.25427.md)）；Seedance 1.5 pro 描述为 SFT → RLHF，蒸馏单列（[来源](https://www.alphaxiv.org/abs/2512.13507.md)） | 教师已经很好，蒸馏只为提速 |
| 在加速模型上做 RL | Seedance 1.0 直接对加速后的 refiner 做 RLHF（[来源](https://arxiv.org/html/2506.09113v2)） | rollout 便宜，适合迭代多轮奖励 |
| 蒸馏和 RL 同时做 | DMDR：RL 与 DMD 并行，4 步学生超过教师（图像，[来源](https://arxiv.org/abs/2511.13649)）；Reward Forcing 的 Re-DMD 按奖励加权（[来源](https://arxiv.org/abs/2512.04678)）；RAVEN 的 CM-GRPO 直接在一致性步上做 RL（[来源](https://arxiv.org/abs/2605.15190)） | 想用奖励补回蒸馏丢掉的运动 / 多样性 |

> [!经验] 顺序的经验法则
> 先 RL 再蒸馏，否则蒸馏学生会丢失 RL 收益（DMD 用的是教师 score，教师没学到的东西学生学不到）。但如果蒸馏教师用的是 RL 前的底模（2026 后训练框架原文写的是“原始预训练模型”，表述不明确），RL 收益要靠之后在少步学生上再做 RL 或 OPD 补回。

### 多样性与运动下降

| 现象 | 原因 | 对策 |
|---|---|---|
| 同一 prompt 出图雷同、物体位置朝向趋同 | DMD 的反向 KL 是 mode-seeking（rCM、CoDMD 都指出） | 一致性目标打底或作正则（rCM、Kling-Omni 保留轨迹损失）；CoDMD 关系正则 |
| 运动变慢、镜头不动、dynamic degree 掉 | 同上；训练中 DMD 的 dynamic degree 会突然塌（CoDMD） | 帧级关系正则；Re-DMD 按运动奖励加权；Phased DMD 分段专家 |
| 多步学生多样性退化到一步水平 | 随机梯度截断（Phased DMD 指出） | 按 SNR 分段训练 |
| 自回归长视频抄首帧 | sink token 过强 | EMA-Sink（Reward Forcing） |

### 音视频联合模型的蒸馏

- **CFG 更多**：文本 guidance、跨模态 guidance、音视频各一套（LTX-2），都要蒸进学生或保留可调。
- **对抗或回归式替代 DMD**：DMAD 在 MiniMax-H3-33B 音视频上明显优于 DMD2 / rCM（[来源](https://arxiv.org/abs/2610.02188)）；SMD 也覆盖了音视频联合生成（[来源](https://arxiv.org/abs/2610.18472)）。
- **流式**：音频 sink token + 非对称块因果对齐（OmniForcing，经 [2026 后训练框架](https://www.alphaxiv.org/abs/2604.25427.md)）；Omni-LiveAvatar 把 LTX-2 渐进蒸馏成少步自回归数字人，单 H200 比教师快 33×（[来源](https://arxiv.org/abs/2608.13602)）。

### 评测

| 维度 | 方法 |
|---|---|
| 整体质量 | VBench Total / Quality / Semantic；人工 GSB（学生 vs 教师） |
| 运动 | VBench dynamic degree；光流幅度 |
| 多样性 | 同 prompt 多 seed 的特征相似度（如 DINO）；目视并排对比 |
| 长视频 | VBench-Long；分段质量漂移（Rolling Forcing 的 ΔDrift） |
| 速度 | 区分 DiT 去噪时间与端到端（文本编码、VAE 解码）；流式看首帧延迟和 FPS，注明 GPU |

**开源工具**：[LightX2V](https://github.com/ModelTC/LightX2V)（步数 + CFG 蒸馏权重与推理）、[FastVideo](https://haoailab.com/blogs/fastvideo_post_training)（DMD、稀疏蒸馏训练）、[rCM](https://github.com/NVlabs/rcm) / [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)、[Causal Forcing](https://github.com/thu-ml/Causal-Forcing)（自回归蒸馏）。

## 选型建议

| 场景 | 推荐配方 |
|---|---|
| 离线高画质、能接受 8–12 步 | CFG 蒸馏 + 分段轨迹蒸馏，再用 TDM / DMD 提质并保留轨迹正则（Kling-Omni、Seedance 路线） |
| 追求 4 步以内、开源底模 | rCM 或 DMD 系（DMD2 / CoDMD）；看重多样性优先 rCM 或带关系正则的 DMD |
| 实时 / 交互 / 世界模型 | 一致性或 ODE 因果初始化 → Self-Forcing DMD → 长 rollout 蒸馏 |
| 音视频联合 | CFG 全部蒸掉；优先 DMAD / SMD 这类不依赖音频 score 估计的方法；流式加 audio sink |
| 已有 RL 后的强教师 | 先 RL 后蒸馏；蒸馏后多样性 / 运动若下降，用 [OPD](#/post-opd) 或奖励加权 DMD 补回 |

## 参考资料

- CFG 蒸馏：[Meng et al. 2022](https://arxiv.org/abs/2210.03142)、[HunyuanVideo v1](https://arxiv.org/abs/2412.03603)
- 一致性 / 轨迹：[LCM](https://arxiv.org/abs/2310.04378)、[PCM](https://arxiv.org/abs/2405.18407)、[Hyper-SD](https://arxiv.org/abs/2404.13686)、[sCM](https://arxiv.org/abs/2410.11081)、[rCM](https://arxiv.org/abs/2510.08431)、[Shortcut](https://arxiv.org/abs/2410.12557)、[MeanFlow](https://arxiv.org/abs/2505.13447)
- 分布匹配：[DMD](https://arxiv.org/abs/2311.18828)、[DMD2](https://arxiv.org/abs/2405.14867)、[TDM](https://arxiv.org/abs/2503.06674)、[SiD](https://arxiv.org/abs/2404.04057)、[Phased DMD](https://arxiv.org/abs/2510.27684)、[CoDMD](https://huggingface.co/papers/2606.21982)、[DMDR](https://arxiv.org/abs/2511.13649)、[SMD](https://arxiv.org/abs/2610.18472)
- 对抗：[ADD](https://arxiv.org/abs/2311.17042)、[LADD](https://arxiv.org/abs/2403.12015)、[Seaweed-APT](https://arxiv.org/abs/2501.08316)、[APT2](https://arxiv.org/abs/2506.09350)、[DMAD](https://arxiv.org/abs/2610.02188)、[RayFlow](https://arxiv.org/abs/2503.07699)
- 自回归：[CausVid](https://arxiv.org/abs/2412.07772)、[Self-Forcing](https://arxiv.org/abs/2506.08009)、[Self-Forcing++](https://arxiv.org/abs/2510.02283)、[LongLive](https://arxiv.org/abs/2509.22622)、[Rolling Forcing](https://arxiv.org/abs/2509.25161)、[Reward Forcing](https://arxiv.org/abs/2512.04678)、[Causal Forcing++](https://arxiv.org/abs/2605.15141)、[Causal-rCM](https://arxiv.org/abs/2606.25473)、[RAVEN](https://arxiv.org/abs/2605.15190)、[MAGI-1](https://arxiv.org/abs/2505.13211)、[Krea Realtime 14B](https://www.krea.ai/blog/krea-realtime-14b)
- 工业报告：[Seedance 1.0](https://arxiv.org/abs/2506.09113)、[Seedance 1.5 pro](https://arxiv.org/abs/2512.13507)、[Seedance 2.0](https://arxiv.org/abs/2604.14148)、[Kling-Omni](https://arxiv.org/abs/2512.16776)、[HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870)、[LTX-2](https://arxiv.org/abs/2601.03233)、[系统性后训练框架](https://arxiv.org/abs/2604.25427)
- 工具：[LightX2V 步数蒸馏文档](https://lightx2v-en.readthedocs.io/en/latest/_sources/method_tutorials/step_distill.md)、[FastVideo 稀疏蒸馏](https://haoailab.com/blogs/fastvideo_post_training)、[TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
