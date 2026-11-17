# 专题：镜头切分

> [!结论] 常用组合
> 视频生成团队做数据时，镜头切分的常见组合是：**TransNetV2 或 PySceneDetect 主切 → embedding 相似度二次校验 → 按时长和画质过滤**。

## 主流算法与工具

| 方案 | 类型 | 特点 | 适用场景 |
|---|---|---|---|
| **PySceneDetect** | 传统方法（开源工具） | 使用最广，Open-Sora、HunyuanVideo 等开源管线默认使用。内置 `ContentDetector`（HSV 帧差）、`AdaptiveDetector`（自适应阈值，减少镜头运动误切）、`ThresholdDetector`（黑场、淡入淡出）、`HistogramDetector`、`HashDetector` | 大规模粗切，纯 CPU，速度快 |
| **TransNetV2** | 深度学习（3D CNN） | 经典的学习型镜头边界检测模型，对硬切和渐变（溶解、淡入淡出）都比较好，GPU 推理快 | 精度要求更高的场景；补 PySceneDetect 对渐变不敏感的短板 |
| **AutoShot** | 深度学习（NAS） | 字节团队在 TransNetV2 基础上的改进，并发布 SHOT 数据集，面向短视频 | 抖音类短视频、特效转场多的素材 |
| **FFmpeg scene 滤镜** | 传统方法 | `select='gt(scene,0.3)'`，无额外依赖 | 快速预处理、精度要求不高 |
| **基于 embedding 的切分 / 合并** | 语义方法 | 用 CLIP / DINOv2 / ImageBind 计算相邻帧相似度，骤降处作为切点；或把语义连续的片段重新拼回 | 二次校验，补漏切、合并碎片 |
| **颜色直方图 + SSIM 分类器** | 传统 + 学习 | Koala-36M 用颜色直方图与结构 SSIM 特征训练 SVM 判断切点 | 需要可解释、轻量的切点判定 |

## 公开报告中的做法

- **Stable Video Diffusion**：在**多个帧率下级联检测切点**。单一帧率容易漏掉渐变转场，几个 fps 下各检测一次再合并。
- **Panda-70M**：先做镜头边界检测，再按语义相似度把过度切碎的片段**重新拼接**，避免产生大量太短的碎片。
- **Waver**：PySceneDetect 之后再用 DINOv2 相邻帧余弦相似度细切，可以抓到渐变和溶解转场。
- **Open-Sora / HunyuanVideo**：用 PySceneDetect 切出单镜头片段，再接美学、运动、OCR 等过滤器。
- **Seedance 1.0**：镜头边界检测（帧间差异或预训练检测器，未说明具体工具），每段最长 12 秒，预训练用 3–12 秒片段，并允许多镜头片段。
- **Kling-Omni**：报告没写切分方法，但在时序质量层**剔除有突兀转场或镜头不连贯的片段**，相当于对切分结果做兜底。

## 落地时最容易出问题的地方

1. **误切**：闪光、爆炸、快速运动、快速摇镜都可能被判成切点。用 `AdaptiveDetector`，或 TransNetV2 + embedding 校验来缓解。
2. **漏切**：溶解、淡入淡出、擦除等渐变转场。TransNetV2 和多帧率检测效果更好。
3. **假单镜头**：画中画、分屏、快剪混剪，切完也不是真单镜头。检查首尾帧 embedding 相似度，过低就丢弃或补切。
4. **切点附近的残帧**：切点前后各裁 2–4 帧，避免片段开头或结尾残留上一个镜头的画面。
5. **多镜头数据**：做多镜头叙事训练时不要把所有切点都切开。保留带切镜的长片段，并在 caption 里标出切点（见 [中训练](#/midtraining)）。

## 推荐管线

```flow
解码 | 可先降到低分辨率 / 低 fps 加速
主切 | TransNetV2 ∪ PySceneDetect AdaptiveDetector
裁边 | 切点前后各裁 2–4 帧
语义校验 | DINOv2 / CLIP 相邻帧相似度：段内骤降补切，首尾差异过大丢弃
过滤 | 时长（如 2–20s）、光流运动分（RAFT / UniMatch）、剔除静止和幻灯片
下游 | 美学打分、OCR / 水印过滤、去重
```

## 选型建议

- **数据量巨大、GPU 紧张**：PySceneDetect `AdaptiveDetector` 跑全量，再对可疑片段用 embedding 校验。
- **追求片段质量**：TransNetV2 主切 + embedding 校验，目前性价比最高。
- **短视频、特效转场多**：试试 AutoShot。

## 最小示例：TransNetV2 + DINOv2 校验

下面是思路示意（伪代码级），用于理解各步骤如何衔接：

```python
# 1) 主切：TransNetV2 输出每帧为切点的概率
predictions = transnet.predict_video(path)          # [num_frames]
cuts = np.where(predictions > 0.5)[0]

# 2) 切点前后裁掉若干帧
segments = [(s + 3, e - 3) for s, e in zip([0, *cuts], [*cuts, n_frames]) if e - s > 8]

# 3) DINOv2 相邻帧校验：段内相似度骤降处补切；首尾差异过大则丢弃
for s, e in segments:
    feats = dinov2(frames[s:e:stride])               # L2 归一化特征
    sim = (feats[1:] * feats[:-1]).sum(-1)
    if sim.min() < 0.6:  # 阈值需在自有数据上标定
        split_at_minimum(...)
    if (feats[0] * feats[-1]).sum() < 0.3:
        drop(...)
```
