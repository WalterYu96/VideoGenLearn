# 专题：视频与图像去重

> [!结论] 分层思路
> 视频去重按相似程度分层处理：**完全相同 → 轻度变换**（转码、缩放、水印）**→ 重度变换**（裁剪、画中画、剪辑拼接）。工业上用“便宜的方法先粗筛，昂贵的方法再精排”。

## 算法分类总览

| 类别 | 代表算法 / 方案 | 视频去重方式 | 能否用于图像 | 抗变换能力 |
|---|---|---|---|---|
| 1. 文件哈希 | MD5、SHA-1、SHA-256 | 对整个文件求哈希 | ✅ | 极弱，只能找字节完全相同的文件 |
| 2. 感知哈希 | aHash、dHash、pHash（DCT）、wHash（小波）、Blockhash | 抽关键帧逐帧求哈希，比较哈希序列 | ✅ 本来就为图像设计 | 抗缩放、压缩、调色；不抗裁剪、旋转 |
| 3. 工业级感知哈希 | Meta **PDQ**（图像）/ **TMK+PDQF**、**vPDQ**（视频）；Microsoft **PhotoDNA** | TMK 对帧特征做时序傅里叶聚合；vPDQ 对逐帧 PDQ 做匹配 | ✅ PDQ、PhotoDNA | 中等，常用于内容安全与审核 |
| 4. 传统局部特征 | SIFT / SURF / ORB + BoVW / VLAD / Fisher Vector + RANSAC 几何校验 | 对关键帧提取特征后检索 | ✅ | 抗裁剪、遮挡、视角变化；计算量大 |
| 5. 深度全局特征 | CNN + R-MAC / GeM；**DINO / DINOv2**、**CLIP**；拷贝检测专用 **SSCD**（Meta） | 帧级 embedding + 时序聚合 | ✅ 当前图像去重主流 | 强，兼顾语义层面的相似 |
| 6. 视频级时空相似度 | **ViSiL**、**DnS**、**S2VS**、TCA、LAMV、Temporal Network、DTW、TransVCL | 帧–帧相似矩阵 + 时序建模 / 对齐，可定位拷贝片段 | ❌ 依赖时序 | 强，能处理剪辑、拼接、片段搬运 |
| 7. 音频指纹 | Chromaprint、Shazam 类 landmark 指纹、Panako | 对音轨做指纹匹配 | ❌ | 抗画面改动，常与视觉特征融合 |
| 8. 多模态融合 | 视觉 + 音频 + OCR / ASR 文本 + 元数据 | 多路召回 + 融合排序 | 部分（视觉 + OCR） | 最强，大平台常用 |

## 可以用于图像去重的方案

- **文件哈希**：只能找完全相同的文件。
- **感知哈希**（pHash、dHash、PDQ、PhotoDNA）：快、存储小，适合海量图片第一道粗筛。
- **局部特征**（SIFT / ORB + RANSAC）：适合检测裁剪、局部抠图后的重复。
- **深度特征**（SSCD、DINOv2、CLIP）：效果最好。SSCD 专门为拷贝检测训练，是 ISC2021 图像相似度挑战赛的代表方法；CLIP / DINOv2 更偏语义近似，可以找出同场景的不同照片。

只能用于视频的是第 6 类（ViSiL、DnS 等时空模型）和第 7 类（音频指纹）。

## 检索与索引

- **二值哈希**：BK-Tree、Multi-Index Hashing、汉明距离 LSH。
- **浮点向量**：**FAISS**（IVF-PQ、HNSW）、**Milvus**、ScaNN。

## 开源工具与数据集

- **工具**：`imagededup`（哈希与 CNN 两种方式）、`ThreatExchange`（Meta，含 PDQ / TMK / vPDQ）、`videohash`、czkawka、dupeGuru、Video Duplicate Finder。
- **视频数据集**：CC_WEB_VIDEO、VCDB、FIVR-200K、SVD（抖音）、VSC2022（Meta 视频相似度挑战赛）。
- **图像数据集**：DISC21（ISC2021）、Copydays、INRIA Holidays。

## 典型工业流水线

```flow
文件 MD5 | 精确去重
关键帧抽取 | 场景切分 / 均匀抽帧
感知哈希粗筛 | pHash / PDQ，汉明距离召回
深度特征召回 | SSCD / DINOv2 / EmbeddingGemma 2 → FAISS / Milvus
视频级精排 | ViSiL / DnS / DTW 时序对齐 + 音频指纹
判重 | 片段定位与阈值
```

图像去重用前几步即可：MD5 → pHash / PDQ 粗筛 → SSCD / DINOv2 精排，需要更高精度时加 SIFT + RANSAC 校验。

**在视频生成数据管线中**：Seedance 1.0 用自研视频 embedding 聚类，每个近重复簇只保留质量最高的一条；Kling-Omni 用帧级 + 时序指纹去重。见 [各家管线对比](#/data-industry)。

## 案例：EmbeddingGemma 2 能否用于去重

EmbeddingGemma 2 属于上表第 5 / 8 类（深度全局特征 + 多模态），与 CLIP 同一路线，但额外支持视频和音频。

| 项目 | 信息 |
|---|---|
| 发布 | Google，2026-10-06，Apache 2.0 |
| 规模 | 740M，基于 Gemma 4 架构；可只加载部分编码器：文本 270M、文本 + 视觉 440M、文本 + 音频 570M、全模态 740M |
| 输出 | 文本 / 图像 / 音频 / 视频统一映射到 768 维；Matryoshka 训练，可截断到 512 / 256 / 128 维（官方称 256 维保留约 95% 检索质量） |
| 上下文 | 所有模态共享 8192 token |
| 使用 | sentence-transformers ≥ 6.1.0；推理用 bf16 或 fp32（不兼容 fp16） |

**视频时长上限（推算，非官方）**：第三方资料称默认每秒抽 1 帧、每帧约 140 token，8192 token 约容纳 58 帧。帧会被缩放到固定尺寸，所以时长基本不受原视频分辨率影响，主要由抽帧率决定：

| 抽帧率 | 单次最长（仅画面） |
|---|---|
| 2 fps | 约 29 秒 |
| 1 fps（默认） | 约 58 秒 |
| 0.5 fps | 约 2 分钟 |
| 0.2 fps | 约 4.8 分钟 |

> [!注意] 用于去重时的局限
> 1. 它是**语义检索模型**，不是拷贝检测模型：两张内容相似但不同源的图也可能得分很高。直接定阈值容易把“相似”判成“重复”。
> 2. 单次约一分钟，长视频仍需切段后各自出向量，且不做片段级时序对齐。
> 3. 缺少第三方在 DISC21 / VCDB / FIVR 上的评测，使用前建议在自有数据上和 SSCD、DINOv2 对比。

**建议定位**：放在“深度特征召回”层（截到 256 维存入 FAISS / Milvus），最终判重交给 SSCD 相似度、SIFT + RANSAC 或视频级时序对齐。长视频按 30–60 秒或按镜头切段，段级比对后再做时序对齐，判断整片重复还是片段搬运。

> [!来源] 参考
> [EmbeddingGemma 2 开发者指南](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/) · [多模态 Embedding 文档](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers) · [The New Stack 报道](https://thenewstack.io/google-embeddinggemma-multimodal-search/) · [aicybr 使用指南](https://aicybr.com/blog/embeddinggemma-2-local-multimodal-search-guide)
