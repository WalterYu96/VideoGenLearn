# 5. 参考生成与多任务能力

> [!结论] 本章结论
> 2026 年的商用旗舰都是 **“All-in-one” 多参考模型**。Seedance 2.0 单次最多吃 9 图 + 3 视频 + 3 音频，Kling-Omni 统一了参考、编辑、镜头续写。技术上只有两种注入方式：**通道拼接**（与目标帧空间对齐的条件）和**序列拼接**（不对齐的参考，作为额外 token 参与注意力）。

## 5.1 能力矩阵

| 参考类型 | 典型任务 | 注入方式 | 代表工作 |
|---|---|---|---|
| 文本 | T2V、多镜头剧本、影视语言控制 | 文本编码器 + Prompt 改写器 | 所有模型；Kling 3.0 多镜头 |
| 图像（对齐） | I2V、首尾帧、关键帧插值 | 通道拼接条件 latent + mask；可叠加 SigLIP / CLIP 特征 | HunyuanVideo 1.5（通道 + SigLIP 序列）、Wan I2V、Waver |
| 图像（非对齐） | 主体 / 人物 / 商品 / 场景一致性（R2V） | 参考 latent 作为额外序列 token，类型 embedding / 独立 RoPE 偏移 | Phantom、SkyReels-A2、MAGREF、Wan 2.7 R2V、Kling 主体库、Sora 2 Cameo |
| 视频 | 续写、编辑、风格迁移、运动 / 运镜迁移、新视角重拍 | 序列拼接（参考视频）或通道拼接（待编辑视频） | Wan VACE、Kling-Omni、Runway Aleph、ReCamMaster、Seedance 2.0 |
| 音频 | 语音驱动数字人、唇形同步、按音乐卡点 | 音频特征（Wav2Vec / Whisper）经 cross-attn 注入；或联合生成时作为音频流条件 | OmniHuman-1.5、Wan 2.2-S2V、InfiniteTalk、MultiTalk |
| 音画联合输出 | 文 → 音视频，含对白、音效、BGM | 双流 DiT + 双向交叉注意力 | Veo 3.x、Sora 2、Seedance 1.5 pro / 2.0、LTX-2、Ovi |
| 姿态 / 骨骼 / 深度 | 动作迁移、角色替换 | 控制信号编码后通道拼接或 ControlNet 式分支 | Wan 2.2-Animate、VACE、UniAnimate |
| 相机 / 3D | 相机轨迹控制、世界漫游、新视角 | Plücker 射线 embedding；或点云 / 深度渲染图作条件 | CameraCtrl、GEN3C、HunyuanWorld-Voyager、Genie 3 |
| 视觉标注 | 箭头 / 框 / 草图指定运动 | 作为图像参考输入，由 MLLM 理解 | Kling-Omni、轨迹控制（DragNUWA / Motion Prompting） |
| 文档 / 网页 | 产品介绍、知识视频 | MLLM 理解后生成剧本与分镜 | Wan 3.0（官方宣传，细节未公开） |

## 5.2 统一多参考模型的训练策略

1. **统一输入格式**：所有条件都编码到同一 latent 空间，用类型 embedding 区分“目标 / 首帧 / 参考主体 / 参考视频 / 音频”；文本中用 `@图1`、`[视频2]` 这样的占位符指代参考。
2. **数据构造是核心难点**：主体参考要“跨片段”采样同一主体（同一人的另一个镜头、去背景后的主体图），否则模型学会“复制粘贴”；编辑数据常用合成对（用已有生成模型或 3D 渲染造前后对）。
3. **条件 dropout**：每种条件独立随机丢弃（Open-Sora 2.0 图像条件 12.5%），支持多条件 CFG；LTX-2 用分模态 CFG（文本和跨模态各一个 scale）。
4. **任务课程**：先 T2V / I2V 打基础，在 CT 阶段注入参考和编辑任务，最后做分任务 SFT / RL（见 [中训练](#/midtraining)）。
5. **参考缓存**：参考 token 不加噪，其 KV 可跨去噪步复用（Kling-Omni 报告约 2× 加速）。

## 5.3 音画联合生成要点

- **架构**：双流（视频流大、音频流小）+ 双向 cross-attn，用时间 RoPE 对齐两个模态的时间轴；音频用独立 VAE（LTX-2：24kHz 立体声 + vocoder）。
- **数据**：只留“有可对应音频”的片段（去掉后配 BGM 且画面无关的）；单独标注对白文本、说话人、语种、音效、音乐。
- **难点**：多人对话口型归属、方言、长时音色一致性；Seedance 2.0 自己列出的局限就包括音频失真和多说话人唇形错误。
