# 4. 后训练：SFT、RL、蒸馏、OPD

> [!结论] 本章结论
> 头部厂商的后训练顺序基本是：**SFT（小而精）→ 偏好 / RL（多奖励）→ Prompt 改写器 RL → 步数 + CFG 蒸馏（可选自回归蒸馏）**。一份 2026-04 的系统性后训练工作报告显示：RLHF 带来 +31% GSB，Prompt 增强再 +20%（[来源](https://www.alphaxiv.org/abs/2604.25427.md)）。

```flow
SFT | 数百到 10 万条类别均衡、人工核验的片段
奖励模型 | 对齐 / 美学 / 运动 / 物理 / 音画同步
偏好与 RL | DPO / ReFL / GRPO 系
Prompt 改写器 RL | 冻结生成器，训练 LLM 改写器
蒸馏 | 步数 + CFG 蒸馏，可选自回归蒸馏
OPD | 学生自采样轨迹上的教师稠密监督
```

## 4.1 SFT

- **数据量极小**：Seedance 1.0 只用数百条类别均衡、人工核验的片段；Waver 720p SFT 约 10 万。
- **做法**：分类别子集各自 SFT 后融合（Seedance 1.0）；合成数据补充美学风格（Waver 人类胜率 32.9% → 39.5%）。
- **定位**：更多是为 RL 提供好的参考策略和接口对齐，而非画质提升主力。

## 4.2 奖励模型

| 奖励维度 | 典型实现 | 使用方 |
|---|---|---|
| 图文 / 文视频对齐 | VLM 打分（Qwen2.5-VL / Qwen3-VL + MLP 头） | HunyuanVideo 1.5、Seedance 1.0、后训练框架 |
| 视觉质量 / 美学 | 关键帧美学 RM、HPSv3 风格排序模型 | Seedance 1.0（keyframe aesthetic RM） |
| 运动质量 | 专门的 motion RM（伪影、形变、幅度） | Seedance 1.0、VideoAlign（VQ / MQ / TA 三维） |
| 几何 / 物理一致 | latent 几何模型预测相机位姿、深度 | [VGGRPO](https://arxiv.org/html/2603.26599)（Google）：相机平滑 + 重投影一致奖励，不需解码 RGB |
| 音画同步 | 口型同步、音质、语音可懂度 | Seedance 1.5 pro 多维 RM（维度未公开） |

> [!经验] 奖励的坑
> 文本对齐是最难的奖励，后训练框架报告其提升最小，原因是 RM 本身不够强。多 RM 加权时要监控各维度，防止“运动幅度变大但画面崩坏”这类 reward hacking。

## 4.3 RL / 偏好优化方法

| 方法 | 类别 | 要点 | 代表应用 |
|---|---|---|---|
| Diffusion-DPO / Flow-DPO | 离线偏好 | 无需采样轨迹，稳定便宜；受限于离线数据 | Kling-Omni（多轮 DPO，专攻运动和画面完整性）、HunyuanVideo 1.5 T2V（约 1 万 prompt） |
| ReFL / 直接奖励反传 | 可微奖励 | 在预测的 $x_0$ 上直接最大化奖励 | Seedance 1.0（称优于 DPO / PPO / GRPO） |
| Flow-GRPO / DanceGRPO | 在线 RL | ODE 转 SDE 引入随机性，组内归一化优势；计算贵 | 学术主流基线，DanceGRPO 已在 HunyuanVideo / Wan 上验证 |
| MixGRPO | 在线 RL | ODE–SDE 混合，只在滑动窗口内做 SDE 与优化，省算力 | HunyuanVideo 1.5（I2V / T2V 在线 RL） |
| 单步 SDE GRPO | 在线 RL | 每组只在同一时刻做一步 SDE（isotemporal grouping），只用终点奖励，去掉 KL | 2026 后训练框架 |
| DiffusionNFT | 在线、无似然 | 在前向过程做正负样本对比，无需存储轨迹，样本效率高 | 可作 GRPO 替代 |

**把 T2I RL 经验迁移到视频时要注意**

1. **采样成本**：一条 720p 5s 视频的 rollout 是一张图的几十倍，所以常在 480p / 短时长 / 少步蒸馏模型上做 RL，再迁移。
2. **轨迹内存**：只对部分时间步做优化（MixGRPO 滑动窗口、单步 SDE）。
3. **Prompt 改写器也做 RL**：冻结生成器，用对齐 + 美学 + 格式奖励训练 LLM（RePrompt 思路；Kling-Omni 的 Prompt Enhancer 也用 RL）。

## 4.4 蒸馏

| 类别 | 代表方法 | 步数 / 收益 |
|---|---|---|
| CFG 蒸馏 | guidance embedding（把 guidance scale 作为条件输入） | 去掉第二次前向，约 2×；HunyuanVideo 1.5 测速即用 CFG 蒸馏模型 |
| 轨迹 / 一致性 | PCM、Hyper-SD TSCD、sCM / rCM（NVIDIA） | rCM 把 Wan 从 100 步压到 3–4 步 |
| 分布匹配 | DMD / DMD2、TDM | 一步到几步；常与轨迹损失一起用作正则 |
| 对抗 | APT、LADD | 提升少步下的细节和锐度 |
| 自回归蒸馏 | CausVid、Self-Forcing、LongLive | 双向教师 → 块因果少步学生，KV cache 流式生成，可实时 |

**工业配方示例**

- **Kling-Omni**：第一阶段轨迹匹配（PCM / Hyper-SD 风格，直接在 10 步调度上训练），第二阶段分布匹配（TDM 思路）并保留轨迹损失作正则；150 NFE → 10 NFE（[来源](https://arxiv.org/html/2512.16776v1)）。
- **Seedance 1.0**：TSCD + score 蒸馏（RayFlow）+ 对抗多步蒸馏（APT），约 10× 加速。
- **后训练框架（2026）**：DMD 少步双向学生 → 加块因果 mask 做 ODE 回归 → Self-Forcing rollout + 视频级 DMD；音视频版加 audio sink token。

> [!提示] 顺序建议
> 先 RL 再蒸馏，否则蒸馏学生会丢失 RL 收益。也有团队直接在蒸馏后的少步模型上做 RL（rollout 便宜）。

## 4.5 OPD：在线策略蒸馏（On-Policy Distillation）

> [!结论] OPD 是什么
> OPD 就是在**学生自己采样的轨迹上，用教师给出稠密监督**。它同时解决了离线蒸馏（SFT）的 exposure bias 和 RL 的稀疏奖励问题。2026 年起它从 LLM（Qwen3、MiMo、GLM-5 的后训练均采用）快速迁移到扩散与视频生成。

**LLM 中的定义**：学生采样 rollout，教师在每个前缀上打分，最小化序列级反向 KL，它可精确分解为逐 token 反向 KL：

$$\mathcal{L}_{OPD} = \mathbb{E}_{y\sim\pi_\theta}\sum_t \Big[\log \pi_\theta(y_t\mid y_{<t}) - \log \pi_T(y_t\mid y_{<t})\Big]$$

[Rethinking OPD](https://arxiv.org/pdf/2604.13016)（清华等，2026-04）的三条发现：

1. 教师要有学生没见过的知识：用 RL 后的 Qwen3 做教师可回收 58.6% 的差距，用基座教师只有 15.6%。
2. 师生的“思维模式”要兼容。
3. 冷启动时先用教师样本做 SFT（该实验用了 20 万条）。

**迁移到扩散 / 视频**：把 “token” 换成“去噪步”或“视频 chunk”，把“教师 logit”换成“教师速度场 / score / 判别器”。

| 方法 | 在谁的轨迹上 | 教师信号 | 损失 | 用途与结果 |
|---|---|---|---|---|
| DMD / DMD2 | 学生少步生成的样本 | 教师 score − 学生分布 fake score | 分布级反向 KL 梯度 | 步数蒸馏；可视为 OPD 的扩散版前身 |
| Self-Forcing | 自回归学生带 KV cache 的完整 rollout | 双向教师 score | 视频级 DMD | 训练与推理一致的流式视频 |
| [DiffusionOPD](https://arxiv.org/html/2605.15055v1)（复旦 + 阿里 Wan 团队，2026-05） | 学生 SDE 轨迹访问到的状态 $x_t$ | 每个任务单独 RL 出的教师速度场 | 每步高斯转移的闭式反向 KL，即 $\lVert\mu_S-\mu_T\rVert^2/2\sigma^2$；ODE 版为 L2 | 多任务 RL 合并：先分任务 RL 出多个教师，再 OPD 到一个学生，避免任务干扰和灾难性遗忘 |
| [AFD 对抗流蒸馏](https://arxiv.org/html/2605.26105v1)（NUS 等，2026-05） | 自回归学生的自条件 rollout | 黑盒教师（Seedance 2.0 API）只给成品视频 | 判别器 Bradley–Terry 优势 + DiffusionNFT 式前向过程更新 + 先验正则 | VBench 物理分 87.55，高于 GAN 83.83、DMD 81.88、SFT 58.69 |
| [OPSD-V](https://huggingface.co/papers/2607.08766)（MeiGen-AI，2026-07） | 学生自回归 rollout 及其自身 KV cache | 同一模型，但用真实长视频替换旧历史的“干净 cache” | 去噪步级稠密自蒸馏 | Self-Forcing / LongLive 长视频；用户研究 66% 偏好 |
| [MILD](https://arxiv.org/abs/2609.34371)（2026-09） | 视频学生 rollout | 图像专家（美学、文字），经线性 connector 对齐 latent | 限制在学生原预测附近的修正 + 光流运动奖励 | 用便宜的图像教师提升视频能力，优于视频教师 OPD 基线 |

**实操要点**

1. **OPD 与 RL 怎么选**：RL 用标量奖励，OPD 用教师的稠密信号。手上有更强的教师（更大模型、RL 后的专家、闭源 API）时，OPD 的样本效率远高于 RL。
2. **典型组合**：按维度 RL 出专家（美学、运动、文字渲染）→ DiffusionOPD 合并 → DMD / rCM 步数蒸馏 → Self-Forcing / OPSD-V 修复长视频误差累积。
3. **冷启动**：先用离线 SFT 或 ODE 回归初始化学生，再进入 OPD。
4. **算力**：每步要跑学生 rollout，再让教师在学生状态上做前向；在少步学生上做 OPD 最划算。
5. **黑盒教师**：只有 API 输出时，用判别器当作隐式奖励（AFD）；在最优判别器假设下，该更新等价于对教师的 on-policy 反向 KL 蒸馏。
6. **与 Z-Image RL 经验最近的是 DiffusionOPD**：它沿用 Flow-GRPO 的 SDE 离散化，只是把奖励换成教师的转移分布。
