# 1. 数据

> [!结论] 本章结论
> 视频数据管线 = **切镜 → 基础过滤 → 打分过滤 → 去重 → caption → 均衡采样 → 分辨率分层**。各家差异主要在“质量模型”和“caption 模型”两处：前者决定画质上限，后者决定指令遵循上限。
>
> 量级上，HunyuanVideo 1.5 从 1000 万+ 小时原始视频切出约 8 亿片段；Waver 在 192p 阶段用 1.85 亿 clip，到 720p SFT 只剩 10 万。

```flow
切镜与预处理 | 镜头边界检测、格式规范化、黑边 / 水印 / 字幕裁剪
基础过滤 | 时长、分辨率、fps、bpp、宽高比、安全
打分过滤 | 美学、清晰度、运动、相机抖动、曝光、AIGC 检测
去重 | 视频 embedding 聚类，每簇保留质量最高的一条
Caption | 稠密时序描述 + 镜头语言 + 多层长度
均衡与分桶 | 语义均衡、计算分桶、离线编码
分辨率分层 | 192p / 256p → 480p → 720p → SFT
```

本章有三个延伸专题：[镜头切分算法](#/data-shot)、[视频与图像去重](#/data-dedup)、[头部团队数据管线对比](#/data-industry)。

## 1.1 切镜与预处理

- **镜头切分**：PySceneDetect（基线）、TransNetV2（深度学习）。Waver 在 PySceneDetect 之后再用 DINOv2 相邻帧余弦相似度细切，可以抓到渐变和溶解转场；Koala-36M 用颜色直方图 + 结构 SSIM 的 SVM 分类器。详见 [镜头切分专题](#/data-shot)。
- **片段长度**：Open-Sora 2.0 保留 2–8s；Waver 2–10s；Seedance 1.0 上限 12s 且允许多镜头。做多镜头叙事时，要刻意保留带切镜的长片段并在 caption 里标出切点。
- **格式规范化**（Open-Sora 2.0）：丢弃 bpp < 0.02、fps < 16、宽高比超出 [1/3, 3] 的片段；统一转码 H.264、长边 ≤ 1080。
- **裁剪**：黑边裁剪；水印 / logo / 字幕区域自适应裁剪（Seedance 1.0 用规则 + 检测模型）。

## 1.2 清洗算子速查

| 算子 | 常用实现 | 作用与经验 |
|---|---|---|
| 文字 / OCR | PaddleOCR、EasyOCR | 按文字框面积占比过滤；Open-Sora 2.0 只统计置信度 > 0.7 的框 |
| 水印 / logo | LAION watermark 分类器、自训检测器 | 优先裁掉而非丢掉，保留数据量 |
| 模糊 / 清晰度 | OpenCV Laplacian 方差、DOVER-technical | 多帧投票（Open-Sora 2.0 取 5 帧多数表决） |
| 运动强度 | RAFT / UniMatch 光流、FFmpeg motion score | 去掉过静态和过剧烈；Waver 只用前景光流，把主体运动和相机运动分开 |
| 相机运动 / 抖动 | 光流统计、相机运动分类器 | 抖动去掉；推拉摇移作为标签写进 caption（HunyuanVideo 1.5 有专门的相机运动识别器） |
| 曝光 / 色彩 | 亮度直方图、质量模型子维度 | Waver 720p 阶段加入过曝、过暗、色彩平衡 |
| 安全 | NSFW / 暴力分类器 | 必备；Seedance 1.0 覆盖色情、暴力、儿童安全 |
| AIGC 检测 | 合成图像检测器 | Wan 2.1 用于去除合成数据，防止模型学到 AI 痕迹 |
| 去重 | CLIP / SigLIP / 自研视频 embedding + 聚类 | Seedance 1.0 每个近重复簇只留质量最高的一条（详见 [去重专题](#/data-dedup)） |
| 特效异常 | 质量模型子维度 | Waver 识别倍速播放、视角突变、“图片动画化”（静图平移伪视频）、主体被裁 |

**工程框架**：大规模预处理通常基于 Ray / Spark + GPU 解码（NVDEC、DALI、decord）；NVIDIA 开源的 Cosmos / NeMo Curator 视频管线可作参考。

## 1.3 质量分与美学分模型

> [!提示] 趋势
> 从“多个单指标阈值”转向“**一个微调过的 VLM 质量模型给出多维缺陷标签**”。

| 模型 | 类型 | 要点 |
|---|---|---|
| LAION Aesthetic Predictor v2 | CLIP + MLP，单帧 | 最常用基线；Open-Sora 2.0 取首 / 中 / 尾 3 帧平均，大部分片段落在 4.5–5.5 |
| Q-Align / DOVER / FAST-VQA | 图像 / 视频质量评估（VQA） | DOVER 分“技术质量”和“美学质量”两支，适合做粗筛 |
| Koala-36M TSA（VTSS） | 3D Swin 动态支 + ConvNeXt 静态支 | 8 位专家对 20 万视频 1–5 打分训练；PLCC 0.897；在双峰分布谷底取阈值 2.5 |
| Waver 质量模型 | VideoLLaMA3 SFT | 100 万+ 标注片段，13 个低质维度；高质量判定人工审核精度 78% |
| Wan 2.1 视觉质量 | 聚类 + 人工 + 专家模型 | 先聚 100 类、每类人工 1–5 分，再训专家模型全量打分；运动分 6 档 |
| VideoScore / VisionReward / VideoAlign | 视频奖励模型 | 主要用于 RL，也可反哺数据筛选 |

**实践建议**（T2I 经验可直接迁移）：

1. 先用聚类分层抽样，每类做少量人工标注，避免质量模型只学到头部类别的审美。
2. 质量分不要只用来过滤：Koala-36M 把运动 / 美学 / 清晰度分数经 AdaLN 注入模型，推理时拉到最高；Open-Sora 2.0 把 motion score 写进 caption。
3. 越往高分辨率阶段，阈值越严、维度越多（Waver：192p 只筛文字和静态 → 480p 8 维 → 720p 12 维）。

## 1.4 Caption

- **模型**：Seedance 1.0 微调 Tarsier2（冻视觉塔、全量调 LM，中英双语）；Waver 基于 Qwen2.5-VL，以 Tarsier2 caption 为输入，给子动作打起止时间戳，并用 DPO 压幻觉；HunyuanVideo 1.5 用 3 个 captioner（图像、结构化视频、I2V 指令式），并用 OPA-DPO 优化。
- **结构**：主体、动作（时序）、环境、光照氛围、镜头语言（运镜 / 景别 / 角度）、风格。Koala-36M 平均约 202 词，Open-Sora 2.0 超过 70% 的 caption > 75 词。
- **多层长度**：同一片段生成短 / 中 / 长 caption 随机采样，兼顾用户的短 prompt。
- **Prompt 改写器**：把用户输入改写成训练 caption 的分布；Seedance 1.0 用 Qwen2.5-14B 做 SFT + LoRA-DPO。
- **音频 caption**：LTX-2 的标注系统覆盖说话人、口音、语种、环境声、音乐。做音画联合生成必须有这一层。

## 1.5 均衡与分桶

- **语义均衡**：Waver 用 Qwen2.5-32B 把 caption 标到 3 级动作体系（12 / 100 / 6000+ 类），尾部类过采样或合成补充；Seedance 1.0 按主体、场景、动作、风格、时长、运动等维度做下采样头部、定向采集尾部。
- **计算分桶**：按帧数 × 分辨率 × 宽高比分桶，每桶单独搜 batch size（Open-Sora 2.0）；Waver 用等 token 长度 batch + 序列打包（SPFHP）。
- **离线编码**：预先算好 text embedding 和 VAE latent，训练时不再解码视频，是提升 MFU 的关键一步。

## 1.6 数据规模参考（Waver）

| 阶段 | 视频 | 图像 |
|---|---|---|
| 192p，12→16fps | 1.85 亿 | 4000 万 |
| 480p，16fps | 4100 万 | 4000 万 |
| 720p，24fps 预训练 | 2200 万 | 1500 万 |
| 720p 继续训练（CT） | 400 万 | 1500 万 |
| 720p SFT | 10 万 | 1500 万 |
| 1080p refiner SFT | 20 万 | 300 万 |

> [!来源] 参考
> [Waver](https://arxiv.org/html/2508.15761) 报告 Table 2；另见 [Seedance 1.0](https://arxiv.org/html/2506.09113)、[Open-Sora 2.0](https://arxiv.org/html/2503.09642)、[Koala-36M](https://arxiv.org/html/2410.08260)、[Wan 2.1](https://arxiv.org/abs/2503.20314)。
