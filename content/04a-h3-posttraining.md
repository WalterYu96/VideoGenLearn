# MiniMax-H3 后训练调研与落地方案

> [!来源] 原文
> 面向：33B 音视频联合生成 DiT（MiniMax-H3，Ref2VA / FL2VA）· 目标：GSB 人工标注 → DPO → 可选在线 RL · 整理日期：2026-11-13。由 `tools/import_survey.py` 从 `H3_posttraining_survey.html` 生成。

<div class="sv-box sv-warn">
<p><strong>阅读说明</strong>：本文的论文结论和数字均为原作者报告，整理时只阅读了摘要、方法和关键实验段落，<strong>没有复现</strong>。2026 年的不少工作还只是 arXiv 预印本。“本地实测”一节的数据来自我们自己在 2×H200 上的运行。“建议值”是基于经验的起点，需要实际扫参确认。</p>
</div>

## 1. 结论速览

<div class="sv-box sv-tldr">
<p><strong>推荐路线：LoRA + 人工 GSB 偏好对 + Flow-DPO（离线），再视效果接 DiffusionNFT（在线 RL）。</strong></p>
<ol>
<li><strong>为什么先做 DPO</strong>：HunyuanVideo 1.5 和 Kling-Omni 都是先用人工偏好做 DPO。HunyuanVideo 给出的原因是“现有奖励模型难以区分细粒度运动质量”。DPO 只需提前生成好的偏好对，训练成本接近普通微调，和你们计划中的 GSB 标注直接对接。</li>
<li><strong>为什么用 Flow-DPO 而不是原版 Diffusion-DPO</strong>：H3 是 rectified flow 模型。原版的 KL 权重随时间步变化，在 flow 模型上会导致奖励作弊和画面伪影；Flow-DPO 改成常数 β 后训练稳定。</li>
<li><strong>为什么只训 LoRA</strong>：33B 全参数训练光优化器状态就约 530 GB。H3 社区目前所有 RL 配方都只支持 LoRA。用 LoRA 还有一个额外好处：DPO 的参考模型直接用“关掉 adapter 的基础模型”，不需要多存一份 33B 权重。</li>
<li><strong>第二阶段为什么选 DiffusionNFT 而不是 GRPO</strong>：H3 已做过 CFG 蒸馏，单条生成很贵（我们实测约 18 分钟）。DiffusionNFT 不需要 CFG，不需要记录采样轨迹，采样器可以任选，可以用 10 步快速采样。<a href="https://verl-omni.readthedocs.io/en/latest/examples/minimax_h3/diffusionnft_trainer_minimax_h3.html">veRL-Omni</a> 已经有 H3 Ref2VA 的 DiffusionNFT LoRA 配方。</li>
<li><strong>最大的风险</strong>：视频越训越静态（运动坍缩）和奖励作弊。需要在标注维度里单独列出“运动”，加入运动类指标，每轮训练后都人工抽检。</li>
</ol>
</div>
<div class="sv-flow">
<span class="sv-n">阶段 0<br>评测集 + 基线 + Best-of-N</span><span class="sv-a">→</span>
<span class="sv-n">阶段 1<br>多 seed 生成 + GSB 分维度标注</span><span class="sv-a">→</span>
<span class="sv-n">阶段 2<br>LoRA Flow-DPO（主力）</span><span class="sv-a">→</span>
<span class="sv-n">阶段 3<br>用标注训练奖励模型</span><span class="sv-a">→</span>
<span class="sv-n">阶段 4<br>在线 DiffusionNFT（可选）</span>
</div>

## 2. H3 现状与约束

### 2.1 模型结构（与后训练相关的部分）

<table>
<tr><th>项目</th><th>内容</th><th>对后训练的影响</th></tr>
<tr><td>主干</td><td>33B dense 单流 DiT，BF16；其中约 13B 参数在 AdaLN 分支</td><td>全参数训练不现实，只能用 LoRA；AdaLN 可以作为单独的低成本微调目标做消融</td></tr>
<tr><td>CFG</td><td>已蒸馏，<code>guidance_scale</code> 固定、<code>negative_prompt</code> 不可用</td><td>依赖 CFG 的 RL 方法（原版 Flow-GRPO）要改造；DPO、DiffusionNFT、RVM 都不需要 CFG</td></tr>
<tr><td>文本编码器</td><td>Qwen3-VL-32B，取第 50 层 hidden states</td><td>训练时必须<strong>预先缓存文本编码</strong>，否则要多加载 32B 参数</td></tr>
<tr><td>输出</td><td>768p、24 fps、32 kHz 立体声，4–15 秒；视频和音频是两个联合去噪的潜变量</td><td>DPO 和 RL 的损失要同时覆盖视频和音频潜变量；音视频的权重分配是一个调参点</td></tr>
<tr><td>检查点</td><td>FL2VA（处理 t2va、fl2va）和 Ref2VA（处理 ref2va）两套</td><td>两套要分别训练 LoRA；我们的数据是 Ref2VA 任务</td></tr>
<tr><td>帧数约束</td><td>17n+5 帧（例如 124 帧约 5.17 秒）</td><td>训练时需要把时长对齐到合法帧数</td></tr>
</table>

### 2.2 本地实测（2×H200，TP2，SGLang 0.5.23）

<table>
<tr><th>场景</th><th>结果</th></tr>
<tr><td>官方 Ref2VA 示例</td><td>端到端 541 秒，去噪 510 秒（约 10.4 秒/步 × 50 步），峰值显存约 82 GB；与官方样例的 PSNR 为 43.8 dB</td></tr>
<tr><td>真实数据 7 条（4–11 秒视频）</td><td>225 / 461 / 656 / 1081 / 1256 / 1426 / 2336 秒，平均约 1063 秒（约 18 分钟），相当于每秒视频需要 56–216 秒</td></tr>
<tr><td>全量 2095 条估算</td><td>单个 2 卡 pod 需要 29–41 天</td></tr>
<tr><td>服务启动</td><td>6–14 分钟（权重加载加预热）</td></tr>
</table>
<p>结论：<strong>采样是整个后训练流程最贵的环节</strong>。每多生成一个候选视频，都要付出约 35 GPU·分钟（768p、50 步）。所以所有方案都要尽量减少生成次数，或者降低生成成本（减少步数、降低分辨率、上 8 卡 Ulysses，参考 <code>text-research/h3/infra/manifests/h3-ref2va-sglang-base-match.yaml</code> 里的 8 卡 Ulysses8 配置）。</p>

### 2.3 随机性与采样参数（与偏好数据多样性相关）

<table>
<tr><th>参数</th><th>说明</th><th>能否用于制造多样性</th></tr>
<tr><td><code>seed</code></td><td>视频和音频各自的生成器用同一个 seed 初始化；条件噪声用 seed 和 seed+1</td><td><span class="sv-tag sv-t-hi">主要来源</span>同条件不同 seed 是标准做法</td></tr>
<tr><td><code>num_outputs_per_prompt</code></td><td>一次请求生成多个，使用 seed+i</td><td>可以，省去重复编码条件的开销</td></tr>
<tr><td><code>num_inference_steps</code> / <code>flow_shift</code> / <code>audio_flow_shift</code></td><td>默认 50 步、12、3</td><td><span class="sv-tag sv-t-mid">谨慎</span>会系统性改变画质，容易让偏好对学到“步数差异”而不是真实偏好</td></tr>
<tr><td><code>imgvid_cond_noise_aug_for_inference</code> / <code>audio_cond_noise_aug_for_inference</code></td><td>默认 0.999 和 1.0（1.0 表示条件不加噪）</td><td><span class="sv-tag sv-t-mid">谨慎</span>改变的是参考一致性，不建议用来制造偏好对</td></tr>
<tr><td>prompt 改写</td><td>同一意图写成多个版本</td><td><span class="sv-tag sv-t-hi">推荐</span>主要用于扩大 prompt 覆盖面；同一组比较对内部的 prompt 保持一致</td></tr>
<tr><td>Rollout SDE（<code>rollout_sde_type</code> sde/cps/ode）</td><td>只支持 t2va，只作用于视频，HTTP API 没有开放</td><td>Ref2VA 用不上；在线 RL 改用 veRL-Omni 自己的 rollout</td></tr>
</table>

### 2.4 现有数据

<ul>
<li><code>data/Online_Sample/s25-2026-11-11-2095.jsonl</code>：2095 条 Ref2VA 样本，都由 Seedance 2.5 生成（字段 <code>video_Seedance25</code>）；参考素材为 <code>image_N.jpg</code> 和 <code>audio_N.m4a</code>；绝大多数为 9:16、720p。</li>
<li>prompt 是 Seedance 的指令式格式（GLOBAL RULES、ACTION TIMELINE 等），<strong>不是 H3 的结构化格式</strong>：中位长度 10,260 字符，其中 1691 条超过 H3-Context-IR 的 7000 字符限制。本文假设已按 H3 格式重写过。</li>
<li>dry-run 结果：默认能直接跑的有 1744 条；330 条有短于 2 秒的音频，约 20 条音频总长超过 15 秒，6 条 Seedance 视频超过 15 秒。</li>
<li>每个 prompt 都有一条 Seedance 2.5 的结果。它可以作为<strong>外部参考样本</strong>（见 <a href="#/h3-rl/s15">GraftOOD</a>），但不适合直接当 DPO 的胜出方（原因见 <a href="#/h3-rl/s21">阶段 1</a>）。</li>
</ul>

## 3. 方法全景

<p>2025–2026 年视频扩散模型的后训练方法大致分为 7 类。下表先给总览，后面逐一展开。</p>
<div class="sv-box sv-tldr">
<p><strong>阅读方式</strong>：本节每个算法都分两层。</p>
<ul>
<li><span class="sv-tag sv-t-hi">简洁版</span>（绿色框，默认显示）：只用 2–4 个大白话的量、一行核心公式和一个数字例子。<strong>读完这一层就能理解算法在做什么。</strong></li>
<li><span class="sv-tag sv-t-info">详细版</span>（蓝色框，默认折叠，点击展开）：开头一张“本卡符号表”，后面是完整的数学推导、论文细节和对 H3 的分析。看公式时忘了符号，随时回到符号表查。</li>
</ul>
<p>几乎所有算法都在调整同一个东西：模型在每个去噪步预测的<strong>“方向”</strong>（论文里叫速度 v）。普通训练（SFT）的做法是：给一个视频加噪，让模型学会“从这个带噪状态往这个视频走”，这个方向下文称为<strong>“答案方向”</strong>。各算法的区别只在于：<strong>拿什么当答案、给答案加多大的正负权重</strong>。</p>
</div>
<table>
<tr><th>类别</th><th>代表方法</th><th>训练信号</th><th>是否在线采样</th><th>成本</th><th>与 H3 匹配度</th></tr>
<tr><td>离线偏好优化</td><td>Flow-DPO、LocalDPO、DynamicsBoost、SyncDPO</td><td>偏好对（人工或自动构造）</td><td>否</td><td>低</td><td><span class="sv-tag sv-t-hi">高</span></td></tr>
<tr><td>在线 DPO</td><td>OnlineVPO、Diffusion-APO</td><td>奖励模型实时排序</td><td>是</td><td>中</td><td><span class="sv-tag sv-t-mid">中</span></td></tr>
<tr><td>奖励直接反传</td><td>ReFL、Seedance 1.0</td><td>可求导奖励模型的梯度</td><td>是</td><td>中，显存高</td><td><span class="sv-tag sv-t-lo">低</span></td></tr>
<tr><td>轨迹类 RL</td><td>Flow-GRPO、DanceGRPO、MixGRPO、Flash-GRPO、LeanGRPO</td><td>组内相对奖励 + 轨迹对数概率</td><td>是</td><td>高</td><td><span class="sv-tag sv-t-lo">低</span></td></tr>
<tr><td>正向过程 RL</td><td>DiffusionNFT、RVM、DiffusionART</td><td>奖励加权的 flow matching 损失</td><td>是</td><td>中</td><td><span class="sv-tag sv-t-hi">高</span></td></tr>
<tr><td>音视频专用</td><td>OmniNFT、AV-GRPO、SyncDPO</td><td>分模态奖励和梯度</td><td>两种都有</td><td>中</td><td><span class="sv-tag sv-t-mid">中</span>（后期优化）</td></tr>
<tr><td>推理时提升</td><td>Best-of-N、Flow-NRG</td><td>奖励模型打分或引导</td><td>不训练</td><td>推理成本 ×N</td><td><span class="sv-tag sv-t-mid">中</span>（作为基线）</td></tr>
</table>

### 3.1 离线偏好优化（DPO 族）

<div class="sv-card">
<h4>Flow-DPO（含 VideoReward / Flow-RWR / Flow-NRG）<span class="sv-tag sv-t-hi">核心推荐</span></h4>
<p class="sv-src">Liu et al., <a href="https://arxiv.org/abs/2501.13918">Improving Video Generation with Human Feedback</a>, NeurIPS 2025（快手可灵团队） · 代码：<a href="https://github.com/KlingAIResearch/VideoAlign">KlingAIResearch/VideoAlign</a></p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：给模型看一对视频（一个好、一个差），让它<strong>在好视频上比原模型进步更多，在差视频上比原模型进步更少</strong>（甚至退步）。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>误差</td><td>给视频加噪后让模型去噪，模型预测的方向和答案方向差多少。误差越小，说明模型越“认可”这个视频，越容易生成它</td></tr>
<tr><td>进步量</td><td>原模型（参考模型）的误差 − 当前模型的误差。正数表示当前模型比原模型更容易生成这个视频</td></tr>
<tr><td>β</td><td>力度旋钮：β 越大越保守，越不容易偏离原模型</td></tr>
</table>
<div class="sv-oneline">得分 = 好视频的进步量 − 差视频的进步量；　损失 = −log σ( β/2 × 得分 )，得分越大损失越小</div>
<p><strong>数字例子</strong>（取 β = 20）：</p>
<ul>
<li>好视频：原模型误差 1.00，当前模型 0.90，进步量 = +0.10；</li>
<li>差视频：原模型误差 1.00，当前模型 1.05，进步量 = −0.05；</li>
<li>得分 = 0.10 − (−0.05) = 0.15，σ(10 × 0.15) = σ(1.5) ≈ 0.82，损失 ≈ 0.20；</li>
<li>对比：如果当前模型和原模型一模一样，得分为 0，σ(0) = 0.5，损失 ≈ 0.69。训练会推着得分变大。</li>
</ul>
<p><strong>Flow-DPO 相对原版 Diffusion-DPO 的唯一关键改动</strong>：理论推导出的力度会随噪声水平变化（噪声越大，力度越接近 0），导致模型在高噪声段放飞自我、出伪影。Flow-DPO 把力度改成<strong>固定的常数 β</strong>，训练就稳定了。</p>
</div>
<details class="sv-deep">
<summary>详细版：数学公式、β<sub>t</sub> 推导与论文细节（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>x<sup>w</sup>, x<sup>l</sup></td><td>胜出（win）和落败（lose）样本的干净 latent</td></tr>
<tr><td>t</td><td>噪声水平，t=0 为干净数据，t=1 为纯噪声</td></tr>
<tr><td>x<sub>t</sub></td><td>加噪后的状态，x<sub>t</sub> = (1−t)·x<sub>0</sub> + t·ε</td></tr>
<tr><td>ε</td><td>加入的高斯噪声</td></tr>
<tr><td>v</td><td>答案方向（速度目标），v = ε − x<sub>0</sub></td></tr>
<tr><td>v<sub>θ</sub>, v<sub>ref</sub></td><td>当前模型、参考模型（原模型）预测的方向</td></tr>
<tr><td>‖v − v<sub>θ</sub>‖²</td><td>“误差”</td></tr>
<tr><td>σ(z)</td><td>sigmoid 函数 1/(1+e<sup>−z</sup>)</td></tr>
<tr><td>β, β<sub>t</sub></td><td>KL 正则系数（力度旋钮），β<sub>t</sub> 是随 t 变化的版本</td></tr>
</table>
<p><strong>问题</strong>：Diffusion-DPO 是为 DDPM（噪声预测）推导的，不能直接用在 rectified flow（速度预测）模型上。</p>
<p><strong>原版 Diffusion-DPO 损失</strong>（ε 为噪声预测；w 表示胜出样本，l 表示落败样本）：</p>
<div class="sv-formula">
L<sub>DD</sub>(θ) = −𝔼 [ log σ ( −(β/2) · ( ‖ε<sup>w</sup> − ε<sub>θ</sub>(x<sub>t</sub><sup>w</sup>,t)‖² − ‖ε<sup>w</sup> − ε<sub>ref</sub>(x<sub>t</sub><sup>w</sup>,t)‖² − ( ‖ε<sup>l</sup> − ε<sub>θ</sub>(x<sub>t</sub><sup>l</sup>,t)‖² − ‖ε<sup>l</sup> − ε<sub>ref</sub>(x<sub>t</sub><sup>l</sup>,t)‖² ) ) ) ]
</div>
<p>利用 ‖ε − ε<sub>pred</sub>‖² = (1−t)² ‖v − v<sub>pred</sub>‖²，可以换成速度形式：</p>
<div class="sv-formula">
L<sub>FD</sub>(θ) = −𝔼 [ log σ ( −(β<sub>t</sub>/2) · ( ‖v<sup>w</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>w</sup>,t)‖² − ‖v<sup>w</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>w</sup>,t)‖² − ( ‖v<sup>l</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>l</sup>,t)‖² − ‖v<sup>l</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>l</sup>,t)‖² ) ) ) ]
</div>
<p><strong>关键发现</strong>：直接推导会得到 β<sub>t</sub> = β(1−t)²。这个权重在高噪声时趋近于 0，导致模型在高噪声段过拟合，出现奖励作弊和伪影。<strong>把 β<sub>t</sub> 改成常数 β 后，所有维度都更稳定。</strong>这是 Flow-DPO 最重要的实践结论。</p>
<details open>
<summary>β<sub>t</sub> = β(1−t)² 中各参数的含义与推导</summary>
<table>
<tr><th>符号</th><th>含义</th></tr>
<tr><td>β</td><td><strong>KL 正则系数</strong>，来自 RLHF 目标“最大化奖励 − β·KL(π<sub>θ</sub> ‖ π<sub>ref</sub>)”。β 越大，模型越保守、越贴近参考模型（基础模型），偏好信号越弱；β 越小，越激进地拟合偏好对，越容易过拟合和被奖励作弊。</td></tr>
<tr><td>t</td><td><strong>rectified flow 的时间，即噪声水平</strong>。论文约定 x<sub>t</sub> = (1−t)·x<sub>0</sub> + t·ε，<strong>t=0 是干净数据，t=1 是纯噪声</strong>；速度目标为 v = ε − x<sub>0</sub>。</td></tr>
<tr><td>(1−t)²</td><td>不是人为设计的权重，而是把损失从“噪声预测误差”改写为“速度预测误差”时自然出现的<strong>换算系数</strong>。</td></tr>
<tr><td>β/2 中的 1/2</td><td>高斯似然比展开时带出的常数，可以并入 β 一起调。</td></tr>
</table>
<p><strong>(1−t)² 的推导</strong>：由 x<sub>t</sub> 和速度可以反推噪声：ε = x<sub>t</sub> + (1−t)·v（代入可验证：(1−t)x<sub>0</sub> + tε + (1−t)(ε − x<sub>0</sub>) = ε）。因此</p>
<div class="sv-formula">
ε − ε<sub>pred</sub> = (1−t)·(v − v<sub>pred</sub>)　⇒　‖ε − ε<sub>pred</sub>‖² = (1−t)²·‖v − v<sub>pred</sub>‖²
</div>
<p>代入 Diffusion-DPO 的损失后，原来的常数 β 就变成了 β<sub>t</sub> = β(1−t)²。从 RLHF 目标出发的完整推导见<a href="#/h3-rl/s28">附录 A</a>。</p>
<p><strong>为什么最后不用它</strong>：t=0（干净数据）时 β<sub>t</sub> = β；t→1（纯噪声）时 β<sub>t</sub>→0。噪声越大，对偏离参考模型的约束越弱，模型会在高噪声的去噪步上过度拟合偏好。这些步决定整体构图和运动，拟合过头就会出现奖励作弊和伪影。论文实验发现，直接改用常数 β，所有维度都更稳定。</p>
<p><strong>落到 H3 的注意事项</strong>：</p>
<ul>
<li><strong>先确认 H3 的时间方向</strong>：如果 H3 实现中 t=0 是噪声、t=1 是数据，推导出的系数会变成 t²。用常数 β 时这一点不影响损失本身，但构造 x<sub>t</sub> 和 v 时的符号必须与 H3 一致。</li>
<li><strong><code>flow_shift</code> 不影响这个公式</strong>：它改变的是 t 的采样分布，也就是训练时各噪声水平出现的频率；训练时最好与 H3 预训练使用的分布一致。</li>
<li><strong>β 的量级需要扫参</strong>：它取决于损失是按元素取平均还是求和、latent 的尺寸等，不能直接照搬论文数值。</li>
</ul>
</details>
<p><strong>直观理解</strong>：损失让当前模型对胜出样本的速度预测误差比参考模型下降得更多，对落败样本的误差比参考模型上升得更多。β 控制允许偏离参考模型的程度。</p>
<p><strong>其他贡献</strong>：</p>
<ul>
<li>Flow-RWR：按奖励加权的回归，实验中效果不如 Flow-DPO。</li>
<li>Flow-NRG：推理时用奖励模型直接在含噪视频上引导，可以给多个目标自定义权重。</li>
<li>论文报告 Flow-DPO 优于 Flow-RWR 和普通 SFT。</li>
</ul>
<p><strong>对 H3</strong>：直接适用。需要注意的是，H3 同时生成视频和音频两个潜变量，损失要分别算两者的速度误差再加权求和。论文没有覆盖音频，音频权重需要自己调。</p>
<p>从 RLHF 目标一步步推到这个损失的完整过程，见<a href="#/h3-rl/s28">附录 A</a>。</p>
</details>
</div>
<div class="sv-card">
<h4>LocalDPO：局部细节偏好优化<span class="sv-tag sv-t-info">补充数据</span></h4>
<p class="sv-src">Huang et al., <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Mind_the_Generative_Details_Direct_Localized_Detail_Preference_Optimization_for_CVPR_2026_paper.pdf">Mind the Generative Details</a>, CVPR 2026 · <a href="https://github.com/1170300714/Local-DPO">代码</a></p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：好样本用真实视频；坏样本是<strong>把真实视频挖掉一块、让模型自己补上</strong>的结果。然后做 DPO，但<strong>只在挖掉的那块区域上算分</strong>。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>好样本</td><td>真实的高质量视频</td></tr>
<tr><td>坏样本</td><td>同一个视频，某个时空区域被挖掉后由基础模型补全（补得通常不如真实的好）</td></tr>
<tr><td>掩码</td><td>被挖掉的那块区域</td></tr>
</table>
<div class="sv-oneline">LocalDPO = Flow-DPO，但误差只在掩码区域里计算</div>
<p><strong>数字例子</strong>：一个视频的 latent 有 1000 个小块，随机挖掉 100 块让模型补。好坏两个样本在另外 900 块上完全一样，算进去只会带来噪声，所以只用那 100 块的误差算 DPO 得分，信号更集中，收敛更快。</p>
<p><strong>优点</strong>：不需要人工标注，也不需要奖励模型，每个视频只推理一次就能造出一对。</p>
</div>
<details class="sv-deep">
<summary>详细版：区域感知损失与论文细节（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>x<sup>w</sup></td><td>真实视频 latent（好样本）</td></tr>
<tr><td>M</td><td>随机时空掩码，被挖掉的位置为 1</td></tr>
<tr><td>x<sup>l</sup></td><td>在掩码区域由冻结基础模型补全的视频（坏样本），掩码外与 x<sup>w</sup> 相同</td></tr>
<tr><td>⊙</td><td>逐元素相乘</td></tr>
<tr><td>其余</td><td>v、v<sub>θ</sub>、v<sub>ref</sub>、β、σ 与 Flow-DPO 卡片相同</td></tr>
</table>
<p><strong>示意形式</strong>（按论文描述写出，不是原文公式）：把 Flow-DPO 损失中的误差项换成只在掩码区域计算的版本：</p>
<div class="sv-formula">
‖v − v<sub>θ</sub>(x<sub>t</sub>, t)‖²　→　‖M ⊙ ( v − v<sub>θ</sub>(x<sub>t</sub>, t) )‖² / ‖M‖<sub>1</sub>
</div>
<ul>
<li><strong>做法</strong>：真实高质量视频作为正样本；用随机时空掩码遮住一部分，再用冻结的基础模型补全，补全结果作为负样本。训练时只在被破坏的区域上计算 DPO 损失（区域感知 DPO）。</li>
<li><strong>优点</strong>：每个 prompt 只需推理一次，不需要外部打分模型或人工标注，收敛快。在 Wan2.1 和 CogVideoX 上提升了保真度、时序连贯性和人类偏好分。</li>
<li><strong>对 H3</strong>：可以作为人工偏好对的廉价补充，提升细节质量。但需要实现“局部补全”能力（可以借助 H3 的条件噪声增强或 inpainting 式的掩码去噪），工程量中等。</li>
</ul>
</details>
</div>
<div class="sv-card">
<h4>DynamicsBoost：基于续写的免标注偏好</h4>
<p class="sv-src">Li et al., <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.pdf">DynamicsBoost</a>, CVPR 2026</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：让模型“续写”真实视频。<strong>真实开头给得越多，模型要自己编的部分越少，结果就越好</strong>，于是不用人工标注就能自动得到好坏配对。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>前缀</td><td>直接给模型的真实开头帧</td></tr>
<tr><td>续写区</td><td>模型自己生成的部分</td></tr>
</table>
<div class="sv-oneline">真实前缀长的版本 = 好样本，前缀短的版本 = 坏样本；只在续写区算 DPO 得分</div>
<p><strong>数字例子</strong>：视频总长 30 帧。</p>
<ul>
<li>好样本：前 20 帧是真实帧，模型续写后 10 帧；</li>
<li>坏样本：前 10 帧是真实帧，模型续写后 20 帧；</li>
<li>第 1–10 帧两者都是真实帧，完全一样，<strong>不计分</strong>；第 11–30 帧计分，并按长度归一化。</li>
</ul>
<p><strong>主要提升</strong>：运动的合理性和时序连贯性。</p>
</div>
<details class="sv-deep">
<summary>详细版：三个区域与非对称 DPO（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>N</td><td>视频总帧数</td></tr>
<tr><td>N<sub>1</sub> &lt; N<sub>2</sub></td><td>两个随机选的真实前缀长度；前缀 N<sub>2</sub> 的样本为胜者 z<sub>0</sub><sup>w</sup>，前缀 N<sub>1</sub> 的为败者 z<sub>0</sub><sup>l</sup></td></tr>
</table>
<p>两个样本可以分成三段（论文原文的划分）：</p>
<ol>
<li><strong>共享前缀</strong>（0 到 N<sub>1</sub>）：两者都是真实帧，相同；</li>
<li><strong>非对称区</strong>（N<sub>1</sub> 到 N<sub>2</sub>）：胜者是真实帧，败者是模型生成的；</li>
<li><strong>全生成区</strong>（N<sub>2</sub> 到 N）：两者都是模型生成的。</li>
</ol>
<p>Asymmetrical Flow-DPO 只在第 2、3 段上累积偏好损失，并按这部分的长度归一化，避免作为条件的前缀帧泄漏虚假的偏好信号。<strong>示意形式</strong>（非原文公式）：在 Flow-DPO 的误差项中乘上“帧序号 &gt; N<sub>1</sub>”的掩码，再除以 N − N<sub>1</sub>。</p>
<ul>
<li><strong>做法</strong>：把模型扩展为续写模型。给定总长度后，用真实帧做前缀、模型生成其余部分；真实前缀越长，生成部分越少，质量越高，由此自然形成偏好顺序（N₂ 帧前缀的样本优于 N₁ 帧前缀的样本，N₁ &lt; N₂）。</li>
<li><strong>Asymmetrical DPO</strong>：共享前缀部分不计损失，只在续写区域计算并按长度归一化，避免条件帧泄漏虚假的偏好信号。</li>
<li><strong>对 H3</strong>：H3 的 FL2VA 支持关键帧条件，理论上可以模拟“前缀”，但这种做法主要提升运动合理性，而且需要真实视频，优先级较低。</li>
</ul>
</details>
</div>
<div class="sv-card">
<h4>SyncDPO：音视频同步偏好<span class="sv-tag sv-t-hi">H3 强相关</span></h4>
<p class="sv-src"><a href="https://arxiv.org/html/2605.12179v1">SyncDPO: Enhancing Temporal Synchronization in Video-Audio Joint Generation via Preference Learning</a>, 2026</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：好样本是音画同步的视频；坏样本是<strong>把同一个视频的音频挪几帧</strong>造出的“对不上口型”的版本。用 DPO 教模型分辨“对齐”和“没对齐”。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>好样本</td><td>原始的音画同步视频</td></tr>
<tr><td>坏样本</td><td>画面不变，音频在时间上平移、错位或局部打乱</td></tr>
<tr><td>错位量</td><td>音频挪了多少；先大后小（课程学习）</td></tr>
</table>
<div class="sv-oneline">坏样本 = (原画面, 挪动后的音频)；用 Flow-DPO 学，错位量逐步从大变小</div>
<p><strong>数字例子</strong>（24 fps）：训练初期把音频挪 0.5 秒（12 帧），模型很容易分辨；后期只挪 2 帧（约 0.08 秒），逼模型学会细微的口型对齐。</p>
<p><strong>为什么需要它</strong>：普通 SFT 的 MSE 损失对“音频早了 0.1 秒”这种细微错位几乎不惩罚，必须明确给出“错位的反例”。</p>
</div>
<details class="sv-deep">
<summary>详细版：负样本构造与课程（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>(x<sub>vid</sub>, x<sub>aud</sub>)</td><td>一个音视频样本的视频 latent 和音频 latent</td></tr>
<tr><td>shift<sub>k</sub>(·)</td><td>把音频在时间轴上平移 k 帧（也可以是错位拼接或局部打乱）</td></tr>
<tr><td>k(step)</td><td>随训练步数递减的错位量</td></tr>
</table>
<p><strong>示意形式</strong>（按论文描述写出，不是原文公式）：</p>
<div class="sv-formula">
x<sup>w</sup> = (x<sub>vid</sub>, x<sub>aud</sub>)，　x<sup>l</sup> = (x<sub>vid</sub>, shift<sub>k(step)</sub>(x<sub>aud</sub>))，　再代入 Flow-DPO 损失
</div>
<p>落到 H3 时，k 要按<strong>音频 latent 的帧率</strong>换算（音频 VAE 有时间压缩，1 个 latent 帧不等于 1 个视频帧）。</p>
<ul>
<li><strong>问题</strong>：音视频联合生成的后训练以 SFT 为主，MSE 损失对细微的时间错位惩罚不足。</li>
<li><strong>做法</strong>：用规则在线构造负样本，即对正样本的音视频做时间扭曲（平移、错位、局部打乱），不需要额外采样或标注。按课程学习从明显错位逐步过渡到细微错位。</li>
<li><strong>对 H3</strong>：非常契合。口型和音效同步是 H3 这类模型的核心卖点，构造成本几乎为零：把胜出样本的音频潜变量平移若干帧，就得到一个“同步更差”的负样本，可以和人工偏好对混合训练。</li>
</ul>
</details>
</div>
<div class="sv-card">
<h4>在线 DPO：OnlineVPO、Diffusion-APO</h4>
<p class="sv-src"><a href="https://arxiv.org/html/2412.15159v1">OnlineVPO</a>（WACV 2026）· <a href="https://arxiv.org/html/2605.07503v1">Diffusion-APO</a>（2026）</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：不提前准备偏好对，而是<strong>边训练边生成</strong>：每一步都用当前模型现生成几个视频，让奖励模型挑出最好和最差的组成一对，马上做一次 DPO。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>当前模型</td><td>正在训练的模型，用它现场生成候选视频</td></tr>
<tr><td>奖励模型</td><td>代替人工当裁判</td></tr>
</table>
<div class="sv-oneline">每一步：生成 N 个 → 奖励模型打分 → 最高分当好样本、最低分当差样本 → 做一次 DPO</div>
<p><strong>数字例子</strong>：某一步生成 4 个视频，得分 6.1、7.3、5.2、6.8，于是 (7.3, 5.2) 组成这一步的偏好对。下一步模型变了，再重新生成。</p>
<p><strong>为什么要在线</strong>：离线偏好对是旧模型生成的；模型训练几轮后，它自己生成的视频已经和这批数据不一样了，旧数据就“过时”了。</p>
</div>
<details class="sv-deep">
<summary>详细版：两种方法的细节（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>π<sub>θ</sub></td><td>当前策略（正在训练的模型）</td></tr>
<tr><td>R<sub>v</sub></td><td>视频奖励模型</td></tr>
<tr><td>ỹ<sub>w</sub>, ỹ<sub>l</sub></td><td>在线选出的胜者和败者</td></tr>
</table>
<p>OnlineVPO 原文的选对方式：每个优化步用 π<sub>θ</sub> 生成 𝒱 = {y<sub>1</sub>, …, y<sub>N</sub>}，取 ỹ<sub>w</sub> = argmax<sub>i</sub> R<sub>v</sub>(y<sub>i</sub>)，ỹ<sub>l</sub> = argmin<sub>i</sub> R<sub>v</sub>(y<sub>i</sub>)，再代入 DPO 损失；参考模型按课程逐步更新，让模型始终在“参考模型附近”往更好的方向走。</p>
<ul>
<li><strong>问题</strong>：离线偏好数据是旧模型生成的。随着训练推进，当前模型的输出分布会偏离这批数据，离线数据逐渐“过时”。</li>
<li><strong>OnlineVPO</strong>：每步用当前模型生成多个视频，由视频质量奖励模型选出最高分和最低分组成偏好对，再做 DPO 更新；参考模型按课程逐步更新。</li>
<li><strong>Diffusion-APO</strong>：同一条件、不同 seed 生成两个候选，由在人工标注上微调过的多模态排序模型判定胜负，然后在线优化。</li>
<li><strong>对 H3</strong>：需要可靠的奖励模型和较快的在线采样。适合放在阶段 3 之后，可作为 DiffusionNFT 的替代方案。</li>
</ul>
</details>
</div>

### 3.2 奖励直接反传（ReFL 类）

<div class="sv-card">
<h4>Seedance 1.0 的奖励反馈学习</h4>
<p class="sv-src"><a href="https://arxiv.org/html/2506.09113v2">Seedance 1.0 技术报告</a>，字节跳动</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：训练时让模型从带噪状态<strong>一步猜出最终视频</strong>，直接交给奖励模型打分，再顺着“分数怎样变高”的梯度去改模型。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>一步预测</td><td>模型根据当前带噪状态，直接猜出的干净视频</td></tr>
<tr><td>奖励模型</td><td>可以求导的打分模型（Seedance 用了 3 个：基础、运动、美学）</td></tr>
<tr><td>权重</td><td>多个奖励的加权系数</td></tr>
</table>
<div class="sv-oneline">损失 = −(各奖励模型对“一步预测视频”打分的加权和)，直接对模型参数求导</div>
<p><strong>数字例子</strong>：三个奖励模型给某个一步预测打分 0.6、0.8、0.5，权重 0.4、0.3、0.3，总分 = 0.24 + 0.24 + 0.15 = 0.63。梯度告诉模型“视频的哪些像素往哪个方向改，总分会升高”，模型就照着改。</p>
<p><strong>代价</strong>：奖励模型必须可以求导；梯度要从奖励模型穿过 VAE 解码器传回 33B 的 DiT，显存开销非常大。</p>
</div>
<details class="sv-deep">
<summary>详细版：公式与报告细节（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>x̂<sub>0</sub></td><td>一步预测的干净 latent；rectified flow 下 x̂<sub>0</sub> = x<sub>t</sub> − t·v<sub>θ</sub>(x<sub>t</sub>, t)</td></tr>
<tr><td>D(·)</td><td>VAE 解码器</td></tr>
<tr><td>R<sub>k</sub>, w<sub>k</sub></td><td>第 k 个奖励模型及其权重</td></tr>
</table>
<p><strong>通用 ReFL 形式</strong>（报告没有给出公式，以下是这类方法的标准写法）：</p>
<div class="sv-formula">
L(θ) = −Σ<sub>k</sub> w<sub>k</sub> · R<sub>k</sub>( D( x̂<sub>0</sub>(x<sub>t</sub>, t; θ) ), c )
</div>
<p>梯度路径为 R<sub>k</sub> → D → x̂<sub>0</sub> → v<sub>θ</sub> → θ。通常只在某个中间时间步做一次一步预测，不需要反传整条去噪链。</p>
<ul>
<li><strong>奖励模型</strong>共 3 个：基础奖励模型（VLM 架构，负责图文对齐和结构稳定性）、运动奖励模型（抑制伪影、鼓励运动幅度）、美学奖励模型（基于关键帧的图像空间模型）。</li>
<li><strong>优化方式</strong>：训练时模拟推理流程，直接预测干净视频 x₀，对多个奖励模型的加权得分求导并最大化。报告称在对比实验中，这种方式比 DPO、PPO、GRPO 更高效、效果更好。</li>
<li><strong>多轮迭代</strong>：扩散模型和奖励模型交替迭代，比动态更新奖励模型更稳定、更可控。</li>
<li>RLHF 还用在了蒸馏后的超分 refiner 上，提升少步数下的运动质量。</li>
<li><strong>对 H3</strong>：要求奖励模型可求导、能接收视频潜变量或解码后的视频，并且要能反传过 VAE 解码器；33B 模型加上 VAE 解码的显存开销很大。需要一整套自研的奖励模型体系，不适合作为起步方案。</li>
</ul>
</details>
</div>

### 3.3 轨迹类 RL（GRPO 族）

<div class="sv-card">
<h4>Flow-GRPO 及其变体（DanceGRPO、MixGRPO、Flash-GRPO、LeanGRPO）</h4>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：同一个 prompt 生成一组视频，<strong>比组平均分高的那条生成路线“加分”，低的“减分”</strong>。“加分”的意思是提高模型再走一遍这条路线的概率，所以必须能算出“模型走出这条路线的概率”。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>优势值 A</td><td>这个视频比组平均好多少（除以组内标准差）</td></tr>
<tr><td>路线概率</td><td>模型在每一步去噪时“恰好走到这一步”的概率。普通 ODE 采样没有随机性、算不出概率，所以要<strong>把采样改成带随机噪声的 SDE</strong>，并记录每一步</td></tr>
</table>
<div class="sv-oneline">损失 = −Σ<sub>每个视频</sub> A × log(模型走出这条路线的概率)</div>
<p><strong>数字例子</strong>：同一 prompt 生成 3 个视频，得分 8、6、4。平均 6，标准差约 1.63，优势值 = +1.22、0、−1.22。第一条路线的概率被调高，第三条被调低，第二条不变。</p>
<p><strong>为什么贵</strong>：每组要生成多条完整路线（4–24 条）；必须用 SDE 采样；要存下每一步，或在训练时重新算一遍。几个变体都是在省这部分成本：MixGRPO 只在部分步用 SDE，Flash-GRPO 每次只训一个时间步，LeanGRPO 省掉重复的前向计算。</p>
</div>
<details class="sv-deep">
<summary>详细版：SDE 采样、GRPO 目标、梯度的本质与各变体（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>G</td><td>组大小（每个 prompt 生成几个）</td></tr>
<tr><td>R<sub>i</sub>, A<sub>i</sub></td><td>第 i 个样本的奖励、优势值，A<sub>i</sub> = (R<sub>i</sub> − mean(R)) / std(R)</td></tr>
<tr><td>g<sub>t</sub></td><td>SDE 注入噪声的强度，g<sub>t</sub> = a·√(t/(1−t))</td></tr>
<tr><td>π<sub>θ</sub>(x<sub>t−Δt</sub> | x<sub>t</sub>)</td><td>单步转移概率（高斯分布）</td></tr>
<tr><td>ρ<sub>t</sub></td><td>新旧策略单步概率之比 π<sub>θ</sub>/π<sub>old</sub></td></tr>
<tr><td>ε<sub>clip</sub></td><td>PPO 式裁剪范围</td></tr>
</table>
<p><strong>第 1 步：ODE 转 SDE。</strong>Flow-GRPO 采用速度参数化下的 SDE（引自 DiffusionNFT 论文 Eq. 2）：</p>
<div class="sv-formula">
dx<sub>t</sub> = [ v<sub>θ</sub>(x<sub>t</sub>, t) + (g<sub>t</sub>²/(2t))·( x<sub>t</sub> + (1−t)·v<sub>θ</sub>(x<sub>t</sub>, t) ) ] dt + g<sub>t</sub> dw<sub>t</sub>
</div>
<p>用 Euler 离散后，每一步都是一个均值依赖 v<sub>θ</sub>、方差为 g<sub>t</sub>²Δt 的高斯分布，因此单步对数概率可以计算。</p>
<p><strong>第 2 步：GRPO 目标</strong>（PPO 式裁剪，这类方法的标准形式）：</p>
<div class="sv-formula">
J(θ) = 𝔼 [ (1/G) Σ<sub>i</sub> (1/T) Σ<sub>t</sub> min( ρ<sub>t</sub><sup>(i)</sup>·A<sub>i</sub>,　clip(ρ<sub>t</sub><sup>(i)</sup>, 1−ε<sub>clip</sub>, 1+ε<sub>clip</sub>)·A<sub>i</sub> ) ] − β·KL(π<sub>θ</sub> ‖ π<sub>ref</sub>)
</div>
<p><strong>第 3 步：梯度的本质</strong>（DiffusionNFT 论文附录 B.3 的分析）：对任意一阶 SDE 采样器，on-policy 的 GRPO 梯度可以化简为</p>
<div class="sv-formula">
∇<sub>θ</sub>L ∝ ∇<sub>θ</sub> [ (1/N) Σ<sub>i</sub> ( A<sup>(i)</sup>·ε<sup>(i)</sup> )<sup>⊤</sup> v<sub>θ</sub>(x<sub>t</sub>, t) ]
</div>
<p>即 GRPO 是在<strong>让速度场对齐“按优势加权的随机噪声”</strong>，时间步和采样器只影响各步的权重。结合 Stein 恒等式，它近似于“奖励对参数的梯度”（剩余的去噪过程不反传），方差比直接反传奖励（ReFL）大，但不用反传整条链。</p>
<p><strong>各变体</strong>：</p>
<table>
<tr><th>方法</th><th>要点</th><th>报告的效果或效率</th></tr>
<tr><td>Flow-GRPO</td><td>ODE 转 SDE；GRPO 适配 flow matching 模型</td><td>原始方法，计算量大</td></tr>
<tr><td>DanceGRPO</td><td>统一扩散和 rectified flow 的 GRPO 框架</td><td>HunyuanVideo + VideoAlign 奖励：画质 4.51→7.03，运动质量 1.37→3.85（转引自 <a href="https://cflabs.ai/research/video/where-open-video-models-fall-short/">CFLabs</a>）</td></tr>
<tr><td>MixGRPO</td><td>ODE 和 SDE 混合采样，只在部分步做 SDE</td><td>HunyuanVideo 1.5 图生视频 RL 阶段采用</td></tr>
<tr><td><a href="https://arxiv.org/html/2605.15980">Flash-GRPO</a></td><td>每次更新只训一个时间步；同组样本使用相同时间步（iso-temporal grouping），去掉时间步难度带来的方差；用时间梯度校正消除时间步相关的梯度缩放</td><td>在 1.3B–14B 模型上，低预算下优于全轨迹训练。论文指出 14B 模型一次 GRPO 实验通常需要数百 GPU 天</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2609.03528.pdf">LeanGRPO</a></td><td>rollout 时就保留计算图，避免更新阶段重复前向</td><td>端到端最高 1.83× 加速，优化目标不变</td></tr>
<tr><td>Euphonium（见 <a href="https://arxiv.org/abs/2603.06623">GRPO 综述</a>）</td><td>把奖励梯度作为过程引导注入每一步，终点奖励和过程奖励结合</td><td>视频任务收敛快 1.66×</td></tr>
</table>
<div class="sv-box sv-bad">
<p><strong>对 H3 的判断</strong>：不建议作为第一选择。原因有三：</p>
<ol>
<li>每组需要 4–8 个 rollout，H3 每个约 35 GPU·分钟；</li>
<li>H3 的 rollout SDE 只支持 t2va，Ref2VA 需要自己实现；</li>
<li>需要存储轨迹或重新前向。</li>
</ol>
<p><a href="https://github.com/X-GenGroup/Flow-Factory/blob/main/examples/grpo/lora/minimax_h3_fl2va/default.yaml">Flow-Factory</a> 有 H3 FL2VA 的 GRPO 配置，但作者明确注明“未在完整检查点上运行验证”。</p>
</div>
</details>
</div>

### 3.4 正向过程 RL（NFT 族）（第二阶段推荐）

<div class="sv-card">
<h4>DiffusionNFT：基于正向过程的在线强化学习</h4>
<p class="sv-src">Zheng, Chen, Ye, Wang, Zhang, Jiang, Su, Ermon, Zhu, Liu. <a href="https://arxiv.org/html/2509.16117">DiffusionNFT: Online Diffusion Reinforcement with Forward Process</a>. ICLR 2026（NVIDIA、清华、斯坦福）· 代码 <a href="https://github.com/NVlabs/DiffusionNFT">NVlabs/DiffusionNFT</a>。NFT 即 <strong>N</strong>egative-aware <strong>F</strong>ine<strong>T</strong>uning（会利用负样本的微调）。</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：就是<strong>带正负号的 SFT</strong>。同一个 prompt 生成几个视频并打分；对好视频，让模型比旧模型<strong>更朝它走</strong>；对差视频，让模型比旧模型<strong>更背离它</strong>。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>旧方向</td><td>旧模型在这个带噪状态下预测的去噪方向（锚点）</td></tr>
<tr><td>答案方向</td><td>普通 SFT 的目标：从这个带噪状态“朝这个视频走”的方向</td></tr>
<tr><td>r</td><td>这个视频的好坏程度，0 到 1：比组平均好则 &gt; 0.5，差则 &lt; 0.5，等于平均则为 0.5</td></tr>
<tr><td>β</td><td>步子大小的倒数：β 越小，每次改得越多</td></tr>
</table>
<div class="sv-oneline">新目标 = 旧方向 + (2r − 1) × (答案方向 − 旧方向) / β；　然后让模型去回归这个新目标（就是普通 MSE）</div>
<p><strong>数字例子</strong>（β = 1，方向简化成一个数）：一组 4 个视频得分 9、7、5、3，组平均 6，与平均的差为 +3、+1、−1、−3，除以尺度 3 后换算成 r ≈ 1、0.67、0.33、0。设某个状态下旧方向 = 10：</p>
<ul>
<li>9 分视频，答案方向 = 14：2r−1 = 1，新目标 = 10 + 1 × (14 − 10) = <strong>14</strong>，完全朝它走；</li>
<li>3 分视频，答案方向 = 12：2r−1 = −1，新目标 = 10 − 1 × (12 − 10) = <strong>8</strong>，反着走；</li>
<li>r = 0.5 的视频：新目标 = 10，等于旧方向，不学。</li>
</ul>
<p><strong>旧模型</strong>：不是固定不变，而是用 EMA 缓慢跟随新模型，起“锚”的作用，防止一步改太多。<strong>这个慢速度要按任务调</strong>，调不好会坍缩，这正是后面 DiffusionART 要解决的问题。</p>
<p><strong>为什么叫“正向过程 RL”</strong>：训练信号来自“往视频上加噪”（正向过程），而不是“模型实际去噪走过的路”（反向过程）。所以它只需要最终的视频，不需要记录生成过程，也不在乎用哪种采样器、多少步生成。</p>
<p><strong>和 GRPO 的区别</strong>：GRPO 要把采样改成 SDE、记录每一步、算每一步的概率；NFT 都不用，训练代码几乎就是 SFT 代码加一个权重 r 和一个旧模型。这是它快 3–25 倍的主要原因。</p>
<p><strong>容易踩的坑</strong>：只用好视频（只有正号）会坍缩，正负两项必须一起用。</p>
</div>
<details class="sv-deep">
<summary>详细版：动机、7 步推导、实现细节、实验、消融、与 GRPO 的关系、落到 H3（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义（对应简洁版）</th></tr>
<tr><td>c</td><td>条件（prompt、参考图等）</td></tr>
<tr><td>x<sub>0</sub>, x<sub>t</sub>, ε</td><td>干净 latent、加噪状态、高斯噪声；x<sub>t</sub> = (1−t)x<sub>0</sub> + tε</td></tr>
<tr><td>v</td><td><strong>答案方向</strong>，v = ε − x<sub>0</sub></td></tr>
<tr><td>v<sup>old</sup></td><td><strong>旧方向</strong>：旧模型（EMA）的速度预测</td></tr>
<tr><td>v<sub>θ</sub></td><td>正在训练的模型的速度预测</td></tr>
<tr><td>r</td><td>好坏程度（“最优概率”），r ∈ [0,1]</td></tr>
<tr><td>π<sup>old</sup>, π<sup>+</sup>, π<sup>−</sup></td><td>旧策略的样本分布，以及按 r 重新加权得到的“正样本分布”和“负样本分布”</td></tr>
<tr><td>v<sup>+</sup>, v<sup>−</sup></td><td>假想的、只在正 / 负样本上训练出的模型的速度</td></tr>
<tr><td>α(x<sub>t</sub>)</td><td>看到 x<sub>t</sub> 时，它来自正样本的概率</td></tr>
<tr><td>Δ</td><td>“强化引导”方向，好模型相对旧模型多出来的那部分速度</td></tr>
<tr><td>d, h</td><td>d = v<sub>θ</sub> − v<sup>old</sup>（模型的改动量）；h = v − v<sup>old</sup>（答案方向 − 旧方向）</td></tr>
<tr><td>η<sub>i</sub></td><td>第 i 步 EMA 的保留率</td></tr>
</table>
<h4>B. 动机：GRPO 类方法的三个问题</h4>
<table>
<tr><th>问题</th><th>具体表现</th><th>NFT 的对应解法</th></tr>
<tr><td>正向不一致</td><td>只在反向采样过程上做优化，破坏了模型与正向扩散过程的对应关系（论文称之为偏离 Fokker-Planck 方程），模型可能退化成“一串级联高斯”</td><td>损失定义在正向过程上，就是标准的扩散损失，天然保持一致</td></tr>
<tr><td>采样器受限</td><td>为了能算概率，必须用一阶 SDE 采样器，用不了 flow 模型默认的 ODE 或高阶求解器</td><td>采样和训练完全解耦，采样器任选（黑盒都行）</td></tr>
<tr><td>CFG 难处理</td><td>RL 时需要同时训练条件和无条件两个模型，复杂且低效</td><td>全程不用 CFG；论文把 CFG 解释为一种“离线的强化引导”（条件模型相当于正、无条件模型相当于负），可以被 RL 学进模型里</td></tr>
</table>
<h4>C. 推导（逐步）</h4>
<p>记号：c 为条件；π<sup>old</sup> 为采样用的旧策略；rectified flow 下 x<sub>t</sub> = (1−t)·x<sub>0</sub> + t·ε，目标速度 v = ε − x<sub>0</sub>（论文 2.1 节约定，与 Flow-DPO 一致，与 DiffusionART 相反）；v<sup>old</sup>、v<sub>θ</sub> 分别为旧模型和训练模型的速度预测。</p>
<p><strong>第 1 步：把奖励变成“最优概率”，把数据想象成正、负两堆。</strong>引入一个二值“最优”事件 o，令 r(x<sub>0</sub>, c) := p(o=1 | x<sub>0</sub>, c) ∈ [0,1]。于是每个生成样本以概率 r 落入正集、以 1−r 落入负集，由 Bayes 公式得到两个分布：</p>
<div class="sv-formula">
π<sup>+</sup>(x<sub>0</sub>|c) = r(x<sub>0</sub>,c)·π<sup>old</sup>(x<sub>0</sub>|c) / p<sub>old</sub>(o=1|c)，　π<sup>−</sup>(x<sub>0</sub>|c) = [1 − r(x<sub>0</sub>,c)]·π<sup>old</sup>(x<sub>0</sub>|c) / [1 − p<sub>old</sub>(o=1|c)]
</div>
<p>显然 π<sup>+</sup> ≻ π<sup>old</sup> ≻ π<sup>−</sup>（按期望奖励排序）。最朴素的改进是直接学 π<sup>+</sup>，即只在正样本上做 SFT（Rejection FineTuning，RFT）。但实验显示<strong>扩散模型只学正样本会坍缩</strong>，这和 LLM 里 RFT 是强基线的情况不同。所以需要把负样本也用起来。</p>
<p><strong>第 2 步：旧分布是正负两堆的混合，这个关系在任何噪声水平上都成立。</strong>由定义直接得到 π<sup>old</sup> = p·π<sup>+</sup> + (1−p)·π<sup>−</sup>，其中 p = p<sub>old</sub>(o=1|c) = 𝔼<sub>π<sup>old</sup></sub>[r]。对两边同时加噪，再用 Bayes 公式，可以得到给定 x<sub>t</sub> 时的后验也是混合（论文 Lemma A.2）：</p>
<div class="sv-formula">
π<sup>old</sup>(x<sub>0</sub>|x<sub>t</sub>,c) = α(x<sub>t</sub>)·π<sup>+</sup>(x<sub>0</sub>|x<sub>t</sub>,c) + [1 − α(x<sub>t</sub>)]·π<sup>−</sup>(x<sub>0</sub>|x<sub>t</sub>,c)，　α(x<sub>t</sub>) = [π<sub>t</sub><sup>+</sup>(x<sub>t</sub>) / π<sub>t</sub><sup>old</sup>(x<sub>t</sub>)]·p
</div>
<p>α(x<sub>t</sub>) ∈ [0,1] 的含义是：<strong>看到带噪状态 x<sub>t</sub> 时，它最终来自正样本的概率</strong>，即 P(o=1 | x<sub>t</sub>)。</p>
<p><strong>第 3 步：速度是后验均值的线性函数，因此三个速度场共线。</strong>最优速度预测满足 v(x<sub>t</sub>) = a<sub>t</sub>·x<sub>t</sub> + b<sub>t</sub>·𝔼[x<sub>0</sub>|x<sub>t</sub>]（a<sub>t</sub>、b<sub>t</sub> 只依赖噪声调度）。把第 2 步的后验混合代入，得到 v<sup>old</sup> = α·v<sup>+</sup> + (1−α)·v<sup>−</sup>。移项即得论文 Theorem 3.1：</p>
<div class="sv-formula">
Δ := α·( v<sup>+</sup> − v<sup>old</sup> ) = (1 − α)·( v<sup>old</sup> − v<sup>−</sup> )
</div>
<p>也就是说，“朝正样本的模型靠拢”和“远离负样本的模型”是<strong>同一个方向</strong>，只是步长不同。论文称 Δ 为<strong>强化引导（reinforcement guidance）</strong>。</p>
<p><strong>第 4 步：定义训练目标。</strong>仿照 CFG 的形式，目标速度为旧速度加上一个引导：</p>
<div class="sv-formula">
v<sup>*</sup>(x<sub>t</sub>, c, t) = v<sup>old</sup>(x<sub>t</sub>, c, t) + (1/β)·Δ(x<sub>t</sub>, c, t)
</div>
<p>1/β 是引导强度。例如取 β = α 时，v<sup>*</sup> = v<sup>+</sup>，就严格保证了策略改进。问题是 v<sup>+</sup>、v<sup>−</sup> 都是未知的，不能直接算 Δ。</p>
<p><strong>第 5 步：隐式参数化，用一个模型同时表示正、负两个策略。</strong>不去单独训练 v<sup>+</sup> 和 v<sup>−</sup>，而是定义：</p>
<div class="sv-formula">
v<sub>θ</sub><sup>+</sup> := (1 − β)·v<sup>old</sup> + β·v<sub>θ</sub>（隐式正策略）　　v<sub>θ</sub><sup>−</sup> := (1 + β)·v<sup>old</sup> − β·v<sub>θ</sub>（隐式负策略）
</div>
<p>两者关于 v<sup>old</sup> 对称。然后让 v<sub>θ</sub><sup>+</sup> 去拟合正样本、v<sub>θ</sub><sup>−</sup> 去拟合负样本，各自都是普通的 flow matching 损失，按 r 和 1−r 加权：</p>
<div class="sv-formula">
L(θ) = 𝔼<sub>c, x<sub>0</sub>∼π<sup>old</sup>, t, ε</sub> [ r·‖v<sub>θ</sub><sup>+</sup>(x<sub>t</sub>, c, t) − v‖² + (1 − r)·‖v<sub>θ</sub><sup>−</sup>(x<sub>t</sub>, c, t) − v‖² ]
</div>
<p><strong>第 6 步：求这个损失的最优解（Theorem 3.2）。</strong>固定 x<sub>t</sub>，记 d = v<sub>θ</sub> − v<sup>old</sup>，则 v<sub>θ</sub><sup>±</sup> = v<sup>old</sup> ± β·d。对 d 求梯度并令其为零：</p>
<div class="sv-formula">
𝔼[ r·(v<sup>old</sup> + βd − v) ] = 𝔼[ (1 − r)·(v<sup>old</sup> − βd − v) ]　（期望对 x<sub>0</sub>|x<sub>t</sub> 取）
</div>
<p>利用 𝔼[r | x<sub>t</sub>] = α、𝔼[r·v | x<sub>t</sub>] = α·v<sup>+</sup>、𝔼[(1−r)·v | x<sub>t</sub>] = (1−α)·v<sup>−</sup>（来自第 2 步的后验分解），整理得：</p>
<div class="sv-formula">
β·d = α·(v<sup>+</sup> − v<sup>old</sup>) + (1 − α)·(v<sup>old</sup> − v<sup>−</sup>) = Δ + Δ = 2Δ　⇒　v<sub>θ*</sub> = v<sup>old</sup> + (2/β)·Δ
</div>
<p>所以<strong>只用一个模型、两个普通的回归损失，就把“强化引导”直接学进了模型参数里</strong>，不需要单独的引导模型，也不需要在采样时做引导。正、负两项缺一不可：只有一项时，最优解里就只剩 α(v<sup>+</sup> − v<sup>old</sup>) 或 (1−α)(v<sup>old</sup> − v<sup>−</sup>) 中的一个，而这两个单项的系数依赖于未知的 α，并且（按 DiffusionART 的分析）会携带不能抵消的偏差。</p>
<p><strong>第 7 步：等价形式，更容易看出“带正负号的 SFT”。</strong>令 h = v − v<sup>old</sup>（这个样本的 flow 目标相对旧模型的差），把第 5 步的损失展开：</p>
<div class="sv-formula">
r·‖βd − h‖² + (1 − r)·‖βd + h‖² = β²‖d‖² − 2β(2r − 1)⟨d, h⟩ + ‖h‖² = β²·‖ v<sub>θ</sub> − v<sub>tgt</sub> ‖² + const
</div>
<div class="sv-formula">
v<sub>tgt</sub> = v<sup>old</sup> + ((2r − 1)/β)·( v − v<sup>old</sup> )
</div>
<ul>
<li>r &gt; 0.5（好于组平均）：目标从旧模型<strong>朝这个样本的 flow 目标</strong>移动；</li>
<li>r &lt; 0.5（差于组平均）：目标<strong>背离</strong>这个样本的 flow 目标；</li>
<li>r = 0.5：目标就是旧模型，不学；</li>
<li>β 越小，移动幅度越大（1/β 是引导强度）；β²‖d‖² 这一项同时起到“别离旧模型太远”的作用。</li>
</ul>
<h4>D. 关键实现细节</h4>
<table>
<tr><th>设计</th><th>做法</th><th>说明</th></tr>
<tr><td>奖励转最优概率</td><td>r = ½ + ½·clip[ (r<sup>raw</sup> − mean<sub>组</sub>(r<sup>raw</sup>)) / Z<sub>c</sub>, −1, 1 ]</td><td>和 GRPO 一样做组内中心化；Z<sub>c</sub> 是归一化尺度，可以用全局奖励标准差。每个 prompt 采样 K 个，用来估计组平均</td></tr>
<tr><td>旧策略软更新（EMA）</td><td>θ<sup>old</sup> ← η<sub>i</sub>·θ<sup>old</sup> + (1 − η<sub>i</sub>)·θ，默认 η<sub>i</sub> = min(0.001·i, 0.5)</td><td>η=0（完全 on-policy）前期快但会严重坍缩；η→1（接近离线）稳定但慢得不实用。最佳做法是先小后大。OCR 任务需要把 η<sub>max</sub> 提到 0.999（单奖励）或 0.95（多奖励）才稳定，说明需要按任务调</td></tr>
<tr><td>自适应损失权重</td><td>把速度损失换成自归一化的 x<sub>0</sub> 回归：‖x<sub>θ</sub> − x<sub>0</sub>‖² / sg(mean(|x<sub>θ</sub> − x<sub>0</sub>|))，其中 x<sub>θ</sub> = x<sub>t</sub> − t·v<sub>θ</sub></td><td>借鉴 DMD，不用手调 w(t)。消融：给大 t（高噪声）更高权重更稳定；反过来（如 w(t) = 1−t）会坍缩。自适应方案稳定匹配或超过手调方案</td></tr>
<tr><td>不用 CFG</td><td>只用条件模型初始化，全程不开 CFG</td><td>SD3.5-M 不开 CFG 时初始 GenEval 只有 0.24，训练后迅速超过开 CFG 的 0.63</td></tr>
<tr><td>训练时间步</td><td>在采样用到的时间步上加噪并计算损失</td><td>附录 C</td></tr>
<tr><td>采样器</td><td>默认二阶 ODE</td><td>样本既用于打分又用于训练，质量很关键。ODE 优于 SDE（PickScore 这类对噪声敏感的奖励尤其明显）</td></tr>
<tr><td>β</td><td>1（稳定）或 0.1（奖励涨得更快）</td><td>β 也是稳定性和速度之间的权衡，接近 1 时稳定</td></tr>
</table>
<h4>E. 算法流程</h4>
<pre><code>初始化：采样策略 v_old ← v_ref，训练策略 v_θ ← v_ref
for 迭代 i:
    # Rollout：数据收集（任意黑盒采样器，只保留最终样本）
    for 每个 prompt c:
        生成 K 个样本 x0^(1..K)，打分 r_raw^(1..K)
        r_norm = r_raw - mean(r_raw)
        r = 0.5 + 0.5 * clip(r_norm / Z_c, -1, 1)
        存入 buffer：(c, x0, r)
    # 训练：标准 flow matching，多一个旧模型和权重 r
    for 每个 minibatch (c, x0, r):
        t, eps ~ 采样时间步与噪声
        x_t = (1-t)*x0 + t*eps ;  v = eps - x0
        v_pos = (1-β)*v_old(x_t,c,t) + β*v_θ(x_t,c,t)
        v_neg = (1+β)*v_old(x_t,c,t) - β*v_θ(x_t,c,t)
        loss = r*||v_pos - v||² + (1-r)*||v_neg - v||²    # 实际用自适应权重版本
        更新 θ
    θ_old ← η_i*θ_old + (1-η_i)*θ ;  清空 buffer</code></pre>
<p>实现提示：LoRA 场景下，v<sub>old</sub> 可以用一份“旧 LoRA 副本”（shadow adapter）实现，不需要复制整个模型。UniRL 的 H3 配方就是用 <code>EmaLoraConfig</code>（rank 64 / alpha 128）维护一个 <code>old</code> 影子 adapter，EMA 采用 warmup 计划（<code>ema_flat_steps: 75</code>，<code>ema_uprate: 0.0075</code>）。</p>
<h4>F. 实验结果（作者报告）</h4>
<p><strong>设置</strong>：SD3.5-Medium（2.5B），512×512，LoRA r=32、α=64，lr 3e-4；每轮 48 组、组大小 24；单奖励对比和消融用 10 步采样，多奖励训练用 40 步；评测统一用 40 步一阶 ODE。</p>
<table>
<tr><th>模型</th><th>迭代</th><th>GenEval</th><th>OCR</th><th>PickScore</th><th>HPSv2.1</th><th>Aesthetic</th><th>ImageReward</th><th>UnifiedReward</th></tr>
<tr><td>SD3.5-M（不开 CFG）</td><td>—</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.204</td><td>5.13</td><td>−0.58</td><td>2.02</td></tr>
<tr><td>+ CFG</td><td>—</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr>
<tr><td>SD3.5-L（8B，开 CFG）</td><td>—</td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr>
<tr><td>FLUX.1-Dev（12B）</td><td>—</td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr>
<tr><td><strong>+ DiffusionNFT（多奖励，不开 CFG）</strong></td><td>1.7k</td><td><strong>0.94</strong></td><td><strong>0.91</strong></td><td><strong>23.80</strong></td><td><strong>0.331</strong></td><td><strong>6.01</strong></td><td><strong>1.49</strong></td><td><strong>3.49</strong></td></tr>
</table>
<ul>
<li><strong>多奖励联合训练</strong>：同时优化 GenEval、OCR、PickScore、ClipScore、HPSv2.1，并且按阶段交替：先在 Pick-a-Pic 上用模型类奖励训练 800 轮提升画质，再交替训练 GenEval、OCR 等规则类奖励。最终这个不开 CFG 的 2.5B 模型，在域内和域外指标上都超过了开 CFG 的 SD3.5-L（8B）和 FLUX.1-Dev（12B）。Aesthetic、ImageReward、UnifiedReward 是训练中没用过的域外奖励。</li>
<li><strong>与 FlowGRPO 单奖励正面对比</strong>：墙钟时间效率高 3–25 倍。GenEval 约 1k 步达到 0.98，而 FlowGRPO 需要超过 5k 步并且开 CFG 才到 0.95。</li>
<li><strong>只训单一奖励会损害泛化</strong>：例如只训 OCR，其他指标会下降（附录 D）。这也说明多奖励组合的重要性。</li>
</ul>
<h4>G. 消融实验结论</h4>
<table>
<tr><th>消融项</th><th>结论</th><th>对 H3 的启示</th></tr>
<tr><td>去掉负样本损失</td><td>奖励<strong>几乎立刻坍缩</strong></td><td>不能退化成“只用好样本做 SFT”（即 RFT / 拒绝采样微调）</td></tr>
<tr><td>采样器</td><td>ODE 优于 SDE；二阶 ODE 在 GenEval 上略好，PickScore 上持平</td><td>直接用 H3 默认的确定性 ODE 采样即可</td></tr>
<tr><td>时间步权重</td><td>高噪声段权重大更稳定；w(t) = 1−t 会坍缩；自适应权重最好</td><td>沿用自适应权重；音视频两个 latent 可能需要分别归一化</td></tr>
<tr><td>EMA 软更新</td><td>η=0 不稳定；η=0.9 太慢；先小后大最好</td><td>EMA 计划需要实测调整，这正是 DiffusionART 指出的痛点</td></tr>
<tr><td>β</td><td>接近 1 稳定；0.1 涨得快</td><td>先用 β=1 跑通，再尝试更小的 β</td></tr>
</table>
<h4>H. 与 GRPO 的关系（附录 B.3）</h4>
<p>论文还从另一个角度分析了 FlowGRPO：对于任意一阶 SDE 采样器，GRPO 的梯度可以化简为 ∇<sub>θ</sub>[ (1/N)·Σ<sub>i</sub> (A<sup>(i)</sup>·ε<sup>(i)</sup>)<sup>⊤</sup>·v<sub>θ</sub> ]。也就是说，<strong>GRPO 本质上是让速度场去对齐“按优势加权的噪声”</strong>；时间步和采样器只影响各步的权重。再结合 Stein 恒等式，这个梯度近似于奖励对参数的梯度（剩余去噪过程不反传）。和直接反传奖励的 ReFL 相比，GRPO 方差更大，但不需要反传整个去噪链。</p>
<table>
<tr><th></th><th>FlowGRPO（反向过程）</th><th>DiffusionNFT（正向过程）</th></tr>
<tr><td>采样器</td><td>必须用一阶 SDE</td><td>任意，推荐 ODE</td></tr>
<tr><td>需要存储</td><td>整条采样轨迹</td><td>只要最终样本和奖励</td></tr>
<tr><td>似然</td><td>需要逐步高斯似然</td><td>不需要（likelihood-free）</td></tr>
<tr><td>CFG</td><td>通常需要</td><td>不需要</td></tr>
<tr><td>on/off-policy</td><td>on-policy，off-policy 需要重要性采样</td><td>天然 off-policy，不需要重要性采样</td></tr>
<tr><td>代码改动</td><td>需要改采样器和训练循环</td><td>在 SFT 代码上加旧模型和 r 加权即可</td></tr>
</table>
<h4>I. 已知局限</h4>
<ul>
<li><strong>依赖 EMA 调参</strong>：论文自己就在 OCR 上把 η<sub>max</sub> 从 0.5 调到 0.999；DiffusionART 把 GenEval 上调好的设置搬到 FLUX 上也不稳定。换到 H3 这种新模型、新任务，大概率需要重新调。</li>
<li><strong>隐藏的一致性偏差</strong>（DiffusionART 的分析，见下一张卡片）：NFT 的目标基于“正向加噪的速度”，与模型实际的反向采样速度存在差距，正负样本只能部分抵消这部分偏差。在 DiffusionART 的 I2V 视频实验中，NFT 较早进入平台期。</li>
<li><strong>不解决奖励作弊</strong>：NFT 只是优化算法，奖励模型的偏差（如偏爱静止视频）照样会被放大，见 <a href="#/h3-rl/s15">3.8 节</a>。</li>
<li><strong>论文只做了图像</strong>：视频和音视频上的效果来自后续工作（veRL-Omni 的 H3 配方、SyncReward、DiffusionART 的对比实验），原论文没有验证。</li>
</ul>
<h4>J. 落到 H3</h4>
<table>
<tr><th>项目</th><th>建议</th></tr>
<tr><td>框架</td><td>veRL-Omni 的 Ref2VA DiffusionNFT LoRA 配方（rank 64，FSDP2 训练加 vLLM-Omni rollout），详见 <a href="#/h3-rl/s18">第 6 节</a></td></tr>
<tr><td>采样</td><td>训练 rollout 用 10 步（配方默认）、低分辨率（Ref2VA 参考图短边 512）；验证用 40 步全分辨率。H3 默认就是确定性 ODE，正好符合“ODE 优于 SDE”的结论</td></tr>
<tr><td>组大小 K</td><td>8 卡配方 <code>ROLLOUT_N=4</code>。H3 生成昂贵，K=4 是合理起点（论文用 24，但那是 2.5B 的图像模型）</td></tr>
<tr><td>β 与 EMA</td><td>先用 β=1、η<sub>i</sub> = min(0.001·i, 0.5)；如果奖励快速上升后不稳定，就提高 η<sub>max</sub>（0.95–0.999）</td></tr>
<tr><td>音视频联合</td><td>损失要同时覆盖视频和音频 latent；veRL-Omni 的 Actor 只对生成的视频和音频行施加 NFT 损失，参考素材行保持固定。自适应权重的归一化建议按模态分别计算，避免一个模态主导（我们的建议，未经验证）</td></tr>
<tr><td>奖励</td><td>配方默认的 CLAP + ImageBind 只衡量音视频语义对齐，<strong>不衡量参考一致性和画质</strong>。需要接入阶段 3 的奖励模型、同步奖励（SyncReward 的论文就在 DiffusionNFT 上验证过）和运动幅度奖励，并且像论文那样多奖励组合，避免只训单一奖励导致泛化下降</td></tr>
<tr><td>监控</td><td>训练奖励、梯度范数（DiffusionART 观察到坍缩前梯度范数会急剧上升）、运动幅度、固定频率的人工抽检</td></tr>
<tr><td>时间方向</td><td>论文约定 v = ε − x<sub>0</sub>（t=1 为噪声）；接入 H3 时要和 H3 的实现核对</td></tr>
</table>
</details>
</div>
<div class="sv-card">
<h4>RVM：基于速度场的奖励匹配</h4>
<p class="sv-src"><a href="https://arxiv.org/abs/2608.23664">Scaling Reinforcement Learning for Diffusion Models via Velocity Matching</a>, 2026</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：和 NFT 同一个思路，但写成更一般的框架：<strong>直接在“方向”上加减</strong>，高分视频的方向加强，低分的减弱，再加一根“别离参考模型太远”的绳子。DiffusionNFT 是它的一个特例。</p>
<div class="sv-oneline">目标方向 ≈ 参考方向 + 力度 × 优势值 × (答案方向 − 参考方向)，再加一个可选的锚定项（示意）</div>
<p><strong>两个最有用的结论</strong>：</p>
<ul>
<li>常规偏好奖励会偏爱<strong>画面干净但几乎不动</strong>的视频。加入运动奖励后，运动明显改善。</li>
<li>方法简化到这一步后，<strong>奖励和锚定怎么设计，比损失的具体形式更重要</strong>。</li>
</ul>
</div>
<details class="sv-deep">
<summary>详细版（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>v<sub>θ</sub>, v<sub>ref</sub></td><td>训练模型、参考模型的速度</td></tr>
<tr><td>v</td><td>答案方向（前向加噪诱导的速度目标）</td></tr>
<tr><td>A</td><td>组内优势值</td></tr>
</table>
<p><strong>示意形式</strong>（按摘要描述写出，不是原文公式）：L = 𝔼 ‖ v<sub>θ</sub> − [ v<sub>ref</sub> + λ·A·( v − v<sub>ref</sub> ) ] ‖² + μ·‖ v<sub>θ</sub> − v<sub>ref</sub> ‖²，其中 λ 是力度，μ 是可选的锚定强度。取 λ = 1/β、A = 2r − 1、μ = 0 即得 DiffusionNFT 的等价目标。</p>
<ul>
<li>认为基于似然的策略梯度对扩散模型并非必要。直接在速度场上强化高奖励方向、抑制低奖励方向，并用可选的锚定项控制相对参考速度的偏移。DiffusionNFT 和 RAM 都是它的特例。</li>
<li>在 Wan2.1-T2V-1.3B 上使用 16 步确定性 ODE、不用 CFG，训练成本比 Flow-GRPO 低 2.2×、比 DanceGRPO 低 11.8×。</li>
<li><strong>重要观察</strong>：常规偏好奖励会偏爱“画面干净但几乎静止”的视频，引入动态跟踪奖励后运动明显改善。另一个结论是，速度更新简化之后，<strong>奖励和锚定项的设计比具体损失形式更重要</strong>。</li>
</ul>
</details>
</div>
<div class="sv-card">
<h4>DiffusionART：先聚合奖励再倾斜<span class="sv-tag sv-t-info">支持成对奖励</span></h4>
<p class="sv-src">Wan, Chen, Zhang, Zhou, Zhao, Kong, Wan, Gai, You. <a href="https://arxiv.org/abs/2610.12572">DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models</a>. arXiv:2610.12572, 2026-10-08（作者含快手可灵团队成员和 NUS 尤洋组）。名字取自 <strong>A</strong>ggregation-first <strong>R</strong>eward <strong>T</strong>ilting（先聚合、再做奖励倾斜）。</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：NFT 给每个视频单独算一个方向，噪声大、容易互相抵消，还要靠调 EMA 来稳住。ART 改成<strong>同一个 prompt 只算一个“好减坏”的方向，所有状态共用</strong>；再用“每批训练两次”自动刹车，不再需要 EMA。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>G（引导方向）</td><td>同一 prompt 下“好视频平均比差视频多出来的方向”，长度归一化成固定大小</td></tr>
<tr><td>缓存方向</td><td>生成视频时模型在每一步实际预测的方向，顺手存下来</td></tr>
<tr><td>漂移</td><td>当前模型相对缓存方向已经改了多少</td></tr>
</table>
<div class="sv-oneline">目标 = 缓存方向 + G；每批数据分两半，各训一次；漂移太大且还在往外推的状态，跳过不训</div>
<p><strong>数字例子：为什么“训练两次”能自动刹车</strong>（方向简化成一个数，G = 1）：</p>
<ul>
<li>第一次训练：目标比当前多 1，模型往前走，比如走了 0.6；</li>
<li>第二次训练：目标仍是“缓存 + 1”，现在只差 0.4，于是只再推 0.4；</li>
<li>如果第一次走过头到 1.2，第二次的差值是 −0.2，<strong>自动往回拉</strong>。不需要调任何参数。</li>
</ul>
<p><strong>只有一对好坏视频时</strong>（K=2）：G 就是“从差视频指向好视频”的方向，所以 GSB 这种成对标注可以直接用。论文中每个 prompt 只生成 2 个视频，GPU 时间约为生成 8 个时的 1/3.4。</p>
<p><strong>效果</strong>：在 HunyuanVideo-13B 上，达到同样画质所需的 GPU 时间比 NFT 少 22 倍。<strong>代价</strong>：要缓存生成过程中每一步的状态和方向，而 NFT 只要最终视频。</p>
</div>
<details class="sv-deep">
<summary>详细版：NFT 的两个病根、三个组件的公式、算法、实验与 H3 适用性（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义（对应简洁版）</th></tr>
<tr><td>u<sub>t</sub></td><td>答案方向，<strong>本论文约定 u<sub>t</sub> = x<sub>0</sub> − ε</strong>，与 Flow-DPO / NFT 的 v 符号相反</td></tr>
<tr><td>s<sub>t</sub></td><td>一个状态 (x<sub>t</sub>, t, c)</td></tr>
<tr><td>v<sub>old</sub>, v<sub>θ</sub><sup>(0)</sup></td><td>rollout 策略的速度；v<sub>θ</sub><sup>(0)</sup> 就是<strong>缓存方向</strong></td></tr>
<tr><td>R<sub>i</sub>, R̄<sub>c</sub></td><td>第 i 个样本的奖励、该 prompt 的平均奖励</td></tr>
<tr><td>r<sub>i</sub>, q</td><td>NFT 中的最优概率；q 是某个状态下的条件期望奖励</td></tr>
<tr><td>g, e</td><td>g：真正提升奖励的方向；e：正向与反向速度场之间的差距（一致性残差）</td></tr>
<tr><td>d, h</td><td>d = v<sub>θ</sub> − v<sub>old</sub>（<strong>漂移</strong>）；h = u<sub>t</sub> − v<sub>old</sub>（样本残差）</td></tr>
<tr><td>G(c)</td><td><strong>引导方向</strong>，奖励与终点的协方差除以其 RMS</td></tr>
<tr><td>m, δ, w</td><td>漂移的 KL 代理、阈值 10<sup>−5</sup>、每个状态的损失权重（含掩码）</td></tr>
<tr><td>K</td><td>每个 prompt 生成的样本数</td></tr>
</table>
<h4>符号约定（注意与 Flow-DPO 方向相反）</h4>
<p>论文的生成方向是 t=1（噪声）到 t=0（干净）；前向状态为 x<sub>t</sub> = (1−t)·x<sub>0</sub> + t·ε，样本速度目标为 <strong>u<sub>t</sub> = x<sub>0</sub> − ε</strong>（指向去噪方向）。这与 Flow-DPO 论文的 v = ε − x<sub>0</sub> <strong>符号相反</strong>，接入 H3 时必须统一成 H3 实现里的约定。s<sub>t</sub> = (x<sub>t</sub>, t, c) 表示一个状态，v<sub>old</sub> 是 rollout 策略的速度预测，v<sub>θ</sub> 是正在训练的模型的速度预测。</p>
<h4>第一部分：诊断 DiffusionNFT 为什么不稳定</h4>
<p><strong>回顾 NFT</strong>：每个 prompt 采样 K 个结果，奖励 R<sub>i</sub> 映射成 r<sub>i</sub> = ½[1 + clip((R<sub>i</sub> − R̄<sub>c</sub>)/Z<sub>c</sub>, −1, 1)] ∈ [0,1]。令 a = (2r−1)/β，NFT 损失等价于向一个冻结目标回归：</p>
<div class="sv-formula">
L<sub>NFT</sub> = β² · 𝔼 ‖ v<sub>θ</sub> − v<sub>tgt</sub> ‖² + const，　v<sub>tgt</sub> = a·u<sub>t</sub> + (1 − a)·v<sub>old</sub>
</div>
<p>也就是说，高奖励样本把速度拉向“前向加噪诱导出的速度 u<sub>t</sub>”，低奖励样本则推离它。</p>
<p><strong>谜题</strong>：理论上正、负两个分支的理想改进方向是共线的（论文 Lemma 1：q·(v<sub>fwd</sub><sup>+</sup> − v<sub>fwd</sub><sup>old</sup>) = (1−q)·(v<sub>fwd</sub><sup>old</sup> − v<sub>fwd</sub><sup>−</sup>) = g，g 为奖励与速度的条件协方差），所以只用其中一个分支就应该够了。但实验中<strong>只用任意一个分支都会坍缩</strong>，必须两个一起用。</p>
<p><strong>根源一：隐藏的“正向-反向一致性力”</strong>。NFT 的目标是用前向加噪构造的，隐含假设“采样器实际使用的速度场 v<sub>rvs</sub><sup>old</sup>”等于“由它自己生成的终点诱导出的理想前向速度 v<sub>fwd</sub><sup>old</sup>”。实际上两者存在差距，记为 e = v<sub>fwd</sub><sup>old</sup> − v<sub>rvs</sub><sup>old</sup>。论文 Proposition 1 证明，在纯 on-policy 点上 NFT 的输出空间作用力为：</p>
<div class="sv-formula">
F<sub>NFT</sub>(s<sub>t</sub>) = 2g + (2q − 1)·e，　其中 F<sub>+</sub> = g + q·e，F<sub>−</sub> = g − (1 − q)·e
</div>
<ul>
<li><strong>2g</strong> 是真正提升奖励的方向；<strong>(2q−1)·e</strong> 是与奖励无关的一致性力，q 为该状态下的条件期望奖励。</li>
<li>正分支以权重 q 把速度拉向 v<sub>fwd</sub><sup>old</sup>，负分支以权重 1−q 把它推开。两个分支在 g 方向上叠加，在 e 方向上相互抵消。这就解释了为什么必须联合训练。</li>
<li>但抵消只有在每个状态都满足 q = ½ 时才精确。prompt 级的奖励中心化只保证 Σ(2r<sub>i</sub>−1) ≈ 0，<strong>保证不了每个状态都平衡</strong>。t=1 时状态里没有终点信息，q = ½，残差为零；噪声越小，状态越能“看出”终点的好坏，q 越极端，残差越大。这个残差会稀释奖励引导，使训练不稳定。</li>
</ul>
<p><strong>根源二：EMA 起作用靠的是“恢复力”，不是 off-policy</strong>。令 d = v<sub>θ</sub> − v<sub>old</sub>（漂移），h = u<sub>t</sub> − v<sub>old</sub>（样本残差），Proposition 2 把 NFT 损失拆成：</p>
<div class="sv-formula">
L<sub>NFT</sub> = 𝔼 [ −2β(2r − 1)·⟨d, h⟩ + β²·‖d‖² ] + const
</div>
<ul>
<li>线性项携带奖励引导和一致性力；二次项 β²‖d‖² 是把 v<sub>θ</sub> 拉回 v<sub>old</sub> 的<strong>恢复力</strong>。</li>
<li>纯 on-policy 时 d = 0，恢复力为零，残差的一致性力没有任何东西去抵抗，于是坍缩。EMA 让参考策略滞后，d ≠ 0，从而产生恢复力。</li>
<li>两个消融支持这个解释：(1) 保留 EMA rollout，但把目标里的 v<sub>old</sub> 换成 v<sub>θ</sub>，仍然坍缩，说明 off-policy 数据本身不起稳定作用；(2) 不用 EMA，每个 rollout batch 拆成两半各更新一次，不再坍缩，但训练<strong>慢了 15.2 倍</strong>。</li>
<li>EMA 的问题在于需要按任务调参：恢复太弱会坍缩，太强又太慢。DiffusionNFT 原论文的默认 EMA 计划在 OCR 任务上就不稳定，需要把保留率提到 0.999；论文作者把 NFT 在 GenEval 上调好的设置直接搬到 FLUX 上，同样出现不稳定甚至坍缩。</li>
<li>一个反直觉的观察：第二次更新时，恢复力的输出空间幅度只有引导的 1/9.4，但聚合后的参数梯度范数却是引导的 4.9 倍。原因是各样本的引导方向差异大，聚合时大量相互抵消，恢复力却不会抵消，于是主导了更新，拖慢训练。</li>
</ul>
<h4>第二部分：DiffusionART 的三个组件</h4>
<p><strong>组件一：先聚合的协方差引导（消除一致性力）</strong>。思路是不再用前向速度 u<sub>t</sub> 构造目标，而是对当前 rollout 分布做指数奖励倾斜 p<sub>τ</sub>(x<sub>0</sub>|c) ∝ p<sub>old</sub>(x<sub>0</sub>|c)·exp(τR)。Lemma 2 表明，倾斜后终点均值的一阶变化正好是奖励与终点的协方差：∂<sub>τ</sub>𝔼<sub>p<sub>τ</sub></sub>[x<sub>0</sub>|c] |<sub>τ=0</sub> = Cov<sub>p<sub>old</sub></sub>(R, x<sub>0</sub>|c)。在 rectified flow 中 ε 与 x<sub>0</sub> 独立，因此这个协方差在所有噪声水平上都等于奖励与 u<sub>t</sub> 的协方差。用同一个 prompt 的 K 个样本估计它，再做 RMS 归一化：</p>
<div class="sv-formula">
Cov̄(R, x<sub>0</sub>|c) = (1/K) Σ<sub>i</sub> (R<sub>i</sub> − R̄<sub>c</sub>)(x<sub>0</sub><sup>(i)</sup> − x̄<sub>0,c</sub>)，　G(c) = Cov̄ / RMS(Cov̄)
</div>
<div class="sv-formula">
v<sub>tgt</sub><sup>ART</sup>(s<sub>t</sub>) = v<sub>θ</sub><sup>(0)</sup>(s<sub>t</sub>) + G(c)　（对该 prompt rollout 中经过的每个状态 s<sub>t</sub>，目标冻结）
</div>
<ul>
<li>直观理解：G(c) 就是“同一 prompt 下，好样本平均比差样本多出来的那个方向”。同一 prompt 的所有状态共享这个方向，每个状态都得到明确的改进方向，而不是一个经由似然比传递的标量优势值。</li>
<li>目标里完全不含前向速度，所以<strong>一致性力从构造上被消除</strong>，不依赖跨样本抵消。</li>
<li>RMS 归一化把引导幅度固定下来，消除不同 prompt、不同奖励模型之间的尺度和方差差异，<strong>不需要引导强度系数</strong>。</li>
<li>v<sub>θ</sub><sup>(0)</sup> 是 <strong>rollout 时缓存的速度预测</strong>；训练是在<strong>反向 ODE 轨迹实际经过的状态</strong>上进行，这一点与 NFT 只用终点不同。</li>
</ul>
<p><strong>组件二：两次更新带来的自适应恢复（替代 EMA）</strong>。每个 rollout batch 拆成两个不相交的 minibatch，各做一次 MSE 更新，目标都是 v<sub>tgt</sub><sup>ART</sup>。令 d = v<sub>θ</sub> − v<sub>θ</sub><sup>(0)</sup>：</p>
<table>
<tr><th></th><th>第一次更新（v<sub>θ</sub> = v<sub>θ</sub><sup>(0)</sup>）</th><th>第二次更新（v<sub>θ</sub> = v<sub>θ</sub><sup>(1)</sup>）</th></tr>
<tr><td>损失</td><td>(w/D)·‖G(c)‖²</td><td>(w/D)·‖G(c) − d‖²</td></tr>
<tr><td>负梯度（输出空间）</td><td>(2w/D)·G(c)：纯引导</td><td>(2w/D)·(G(c) − d)：引导 + 恢复</td></tr>
</table>
<p>恢复项 −d 的大小与每个状态的漂移成正比：漂移小时引导占主导，漂移大时自动拉回。<strong>不需要任何额外的调参</strong>。由于 G(c) 在同一 prompt 内共享，引导之间的相互抵消大幅减少，恢复力不会像 NFT 那样主导更新，因此不会牺牲速度。消融显示，只做一次更新的 ART 学得快但会逐渐退化，加上第二次更新后能长期稳定。</p>
<p><strong>组件三：ODE rollout 的信任域掩码</strong>。两次更新都带引导梯度，单个 batch 内仍可能更新过大。ODE 转移是确定性的，没有有限的 KL 可以用来度量策略变化，于是借用 Flow-GRPO 的 ODE 转 SDE 公式，构造参考高斯转移（rollout 本身仍是 ODE），定义漂移的 KL 代理：</p>
<div class="sv-formula">
m(s<sub>t</sub>) = (1/D)·KL(q<sub>θ</sub> ‖ q<sub>old</sub>) = ‖μ(s<sub>t</sub>, v<sub>θ</sub>) − μ(s<sub>t</sub>, v<sub>θ</sub><sup>(0)</sup>)‖² / (2D·σ<sub>ref</sub>²(t))，　σ<sub>ref</sub>²(t) = ν²·t/(1−t)·h
</div>
<div class="sv-formula">
w(s<sub>t</sub>) = t · [ 1 − 𝟙{m(s<sub>t</sub>) ≥ δ} · 𝟙{⟨d, G(c) − d⟩ &gt; 0} ]，　δ = 10<sup>−5</sup>（全文固定）
</div>
<ul>
<li>μ(s<sub>t</sub>, v) 是用速度 v 走一步参考 SDE 后的均值，h 是步长，ν 是参考噪声强度；σ<sub>ref</sub> 与策略无关，所以 m 实际上就是“速度漂移 d 的缩放平方范数”（附录 C.2：μ 之差 = h(1+ν²/2)·d）。</li>
<li>只有“漂移已经超过阈值”并且“本次更新还在往外推”时，才把该状态的损失屏蔽掉；往回拉的更新始终保留（借鉴 Flow-DPPO 的思路）。</li>
<li>前面的因子 t 用来加大高噪声状态的权重。</li>
<li>消融：去掉掩码后，T2V 训练在约第 200 步出现一次明显回落再恢复；加上掩码则稳步提升。</li>
</ul>
<h4>算法流程（一次迭代）</h4>
<pre><code># Rollout 阶段
用 ODE 求解器、当前模型 v_θ 采样（v_θ^(0) = v_θ）
对每个 prompt c：收集 K 个终点 x0^(i) 及奖励 R_i、轨迹状态 s_t、缓存的速度预测 v_θ^(0)(s_t)
G(c) = normalize_rms( mean_i[(R_i - R̄_c) * (x0^(i) - x̄0_c)] )
对 c 的每个状态冻结目标：v_tgt(s_t) = v_θ^(0)(s_t) + G(c)
&nbsp;
# Training 阶段
把样本分成两个等大、不相交的 minibatch B1、B2
for B in (B1, B2):
    重新计算每个状态的 w(s_t)（含信任域掩码）
    loss = mean_{s_t∈B} w(s_t) * MSE(v_θ(s_t), v_tgt(s_t))
    做一步优化</code></pre>
<h4>成对奖励扩展（DiffusionART-Pairwise）</h4>
<ul>
<li>把每个 prompt 的 K 个终点分成 K/2 个不相交的配对，由成对奖励模型判定胜负：胜者 y=1、败者 y=0、平局双方都为 ½。用 y 替代标量奖励 R，其余流程完全不变。</li>
<li>论文附录 C.3 证明，此时协方差等于各非平局配对的“胜者减败者”之和乘一个常数（平局配对贡献为零）。<strong>K=2 时，归一化后的引导就是从差样本指向好样本的方向</strong>。</li>
<li><strong>要按 prompt 汇总</strong>：同一 prompt 的多个配对应先汇总协方差再归一化（prompt-global），不要每对单独归一化（pair-local）。GenEval 上前者 0.91、后者 0.78。</li>
<li>对 GSB 的意义：GSB 标注本身就是成对的，用它训练出的成对奖励模型可以直接用于在线训练，不必先转成标量分数。</li>
</ul>
<h4>实验设置与结果</h4>
<table>
<tr><th>任务</th><th>设置</th><th>结果（作者报告）</th></tr>
<tr><td>T2I：SD3.5-M</td><td>LoRA r=32、α=64，512×512，lr 3e-4，10 步 rollout，无 CFG，无 KL</td><td>达到 Flow-GRPO 第 500 步的奖励，所需步数在 GenEval、OCR、PickScore 上分别少 10.0×、5.2×、9.6×；不调 EMA 就避免了 on-policy NFT 的坍缩，效率与精调过 EMA 的 NFT 相当</td></tr>
<tr><td>T2I：FLUX.1-dev（12B）</td><td>6 步 rollout，内置引导嵌入 3.5</td><td>三种方法中学得最快；照搬 SD3.5 设置的 NFT 奖励剧烈波动</td></tr>
<tr><td><strong>T2V：HunyuanVideo-13B（CFG 蒸馏）</strong></td><td><strong>全参数</strong>，lr 1e-5，无 KL；每轮 32 个 prompt × 8 个视频；53 帧、480×480、8 FPS、16 步；VideoReward 打分，TA:VQ:MQ = 7:2:1</td><td>VQ 达到 5.5：24 步、63.4 GPU·小时，对比 NFT(EMA) 273 步、1401.4 GPU·小时（<strong>步数少 11.4×，GPU 时间少 22.1×</strong>）。第 400 步评测 VQ 8.970、MQ 4.224，比最强基线分别高 31.4% 和 57.4%。VQ:MQ = 1:1 设置下，达到 NFT 和 Flow-GRPO-Fast 后期平均奖励的 GPU 时间分别少 12.02× 和 11.72×</td></tr>
<tr><td>I2V：Wan2.1-I2V-14B（CFG 和步数都已蒸馏）</td><td>全参数，lr 1e-5；32 个 prompt × 8 个视频；53 帧、400×640、<strong>4 步</strong>；VQ:MQ = 1:1</td><td>MQ 达到 6：45.8 GPU·小时，对比 Flow-GRPO-Fast 220.3（少 4.8×）；NFT 很早就进入平台期</td></tr>
</table>
<p><strong>成对反馈与组大小</strong>（T2V，400 步，以标量 VQ 作为评测真值）：</p>
<table>
<tr><th>设置</th><th>GPU·小时</th><th>VQ</th></tr>
<tr><td>标量奖励，K=8</td><td>891.1</td><td>13.637</td></tr>
<tr><td>成对奖励，K=2</td><td><strong>260.8</strong></td><td>12.235</td></tr>
<tr><td>成对奖励，K=4</td><td>465.6</td><td>10.883</td></tr>
<tr><td>成对奖励，K=8</td><td>883.3</td><td>12.092</td></tr>
</table>
<p>结论：成对反馈的效果接近标量反馈；对组大小不敏感，<strong>每个 prompt 一对（K=2）就够了</strong>，GPU 时间约为 K=8 的 1/3.4。</p>
<h4>对 H3 的适用性分析</h4>
<table>
<tr><th>方面</th><th>判断</th></tr>
<tr><td>CFG 蒸馏</td><td><span class="sv-tag sv-t-hi">匹配</span>论文的两个视频实验用的正是 CFG 蒸馏模型（HunyuanVideo）以及 CFG 加步数都蒸馏过的模型（Wan2.1 LightX2V），全程不用 CFG</td></tr>
<tr><td>采样成本</td><td><span class="sv-tag sv-t-hi">匹配</span>K=2 就有效，每个 prompt 只需生成 2 个视频，正好缓解 H3 生成昂贵的问题</td></tr>
<tr><td>成对奖励</td><td><span class="sv-tag sv-t-hi">匹配</span>可以直接使用阶段 3 用 GSB 训练的成对奖励模型</td></tr>
<tr><td>需要缓存轨迹</td><td><span class="sv-tag sv-t-mid">额外工程</span>与 NFT 只需终点不同，ART 需要 rollout 返回<strong>每一步的状态 x<sub>t</sub> 和速度预测 v<sub>θ</sub><sup>(0)</sup></strong>。veRL-Omni 的 H3 rollout 是为 NFT 设计的，可能只返回终点，需要改造。好在 10–16 步采样时，缓存的 latent 相比模型权重很小</td></tr>
<tr><td>音视频联合</td><td><span class="sv-tag sv-t-mid">需要设计</span>论文只有视频。H3 的 x<sub>0</sub> 包含视频和音频两个 latent，G(c) 可以在拼接后的 latent 上计算，也可以对两个模态分别做 RMS 归一化，以防一个模态主导方向（后者是我们的推测，论文没有验证）。不同奖励作用于不同模态时，可以借鉴 OmniNFT 按模态计算各自的 G</td></tr>
<tr><td>LoRA vs 全参数</td><td><span class="sv-tag sv-t-mid">未验证</span>视频实验都是全参数（13B/14B），T2I 用的是 LoRA。H3 33B 只能用 LoRA，效果需要实测</td></tr>
<tr><td>时间方向</td><td><span class="sv-tag sv-t-mid">注意</span>论文 u<sub>t</sub> = x<sub>0</sub> − ε，与 Flow-DPO 符号相反，要与 H3 实现对齐；G(c) 是加在速度上的，符号错了会朝反方向优化</td></tr>
<tr><td>奖励作弊</td><td><span class="sv-tag sv-t-lo">依然存在</span>ART 解决的是优化稳定性，而不是奖励模型本身的偏差。运动坍缩和奖励作弊的防护（见 3.8 节）仍然需要</td></tr>
<tr><td>现成实现</td><td><span class="sv-tag sv-t-lo">没有</span>目前没有 H3 实现。算法本身很简单，在 veRL-Omni 的 NFT 框架上改造的主要工作是：rollout 返回轨迹、目标计算、两次更新和掩码</td></tr>
</table>
<div class="sv-box sv-ok">
<p><strong>一个可以尝试的思路（我们的推测，论文没有这样做）</strong>：H3 的默认采样是确定性 ODE，给定条件、seed 和步数，轨迹可以<strong>完全重放</strong>。因此阶段 1 中由基础模型生成的 GSB 人工配对，只要记录了 seed 和步数，就可以重新跑一遍拿到 x<sub>t</sub> 和 v<sup>(0)</sup>，再按 K=2 的 Pairwise 公式做一次 ART 更新。不过这只在第 0 轮严格 on-policy，模型更新后就会变成 off-policy。所以它更适合作为“用人工配对热启动”的手段，与 Flow-DPO 做对比实验，不能替代在线训练。</p>
</div>
</details>
</div>

### 3.5 音视频联合生成专用方法

<div class="sv-card">
<h4>OmniNFT、AV-GRPO：按模态分开算账</h4>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：音视频一起训练时，如果只给整个视频打一个总分，<strong>画面好、声音差</strong>的样本会得到“中等”分，两个模态都学不到东西。解决办法是<strong>画面的分只用来训画面，声音的分只用来训声音</strong>。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>视频优势值</td><td>这个样本的画面比组平均好多少</td></tr>
<tr><td>音频优势值</td><td>这个样本的声音比组平均好多少</td></tr>
</table>
<div class="sv-oneline">总损失 = 视频优势值 × 视频部分的损失 + 音频优势值 × 音频部分的损失（同步奖励两边都算）</div>
<p><strong>数字例子</strong>：某个样本画面比平均好 +1，声音比平均差 −1。</p>
<ul>
<li>只用总分：(+1) + (−1) = 0，这个样本<strong>完全被浪费</strong>；</li>
<li>按模态分开：画面那部分按 +1 往好的方向学，声音那部分按 −1 往反方向学，两个信息都用上了。</li>
</ul>
<p><strong>对 H3</strong>：H3 是单流 DiT，没有“视频塔、音频塔”之分，所以“冻结一个塔”这类做法不能直接照搬。但“按模态分开算账”可以做：在损失里把视频 latent 和音频 latent 两部分分开加权即可。</p>
</div>
<details class="sv-deep">
<summary>详细版：三种方法对比与 H3 启示（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>A<sup>vid</sup>, A<sup>aud</sup>, A<sup>sync</sup></td><td>分别用视频奖励、音频奖励、同步奖励算出的组内优势值</td></tr>
<tr><td>L<sup>vid</sup>, L<sup>aud</sup></td><td>只在视频 latent / 音频 latent 上计算的逐样本损失</td></tr>
</table>
<p><strong>示意形式</strong>（按 OmniNFT 的“按模态分配优势值”思想写出，不是原文公式）：</p>
<div class="sv-formula">
L = Σ<sub>i</sub> [ (A<sub>i</sub><sup>vid</sup> + A<sub>i</sub><sup>sync</sup>)·L<sub>i</sub><sup>vid</sup> + (A<sub>i</sub><sup>aud</sup> + A<sub>i</sub><sup>sync</sup>)·L<sub>i</sub><sup>aud</sup> ]
</div>
<p>对 NFT 类方法，就是把 r 换成按模态的 r<sup>vid</sup>、r<sup>aud</sup>，分别作用于视频行和音频行的速度回归。</p>
<table>
<tr><th>方法</th><th>要解决的问题</th><th>做法</th><th>基座</th></tr>
<tr><td><a href="https://arxiv.org/html/2605.12480">OmniNFT</a></td><td>(1) 同一组内不同模态的优势值不一致；(2) 视频分支的梯度渗入浅层音频层；(3) 同步关键区域得不到充分探索</td><td>按模态分配优势值（每种奖励只作用于对应模态分支）；逐层梯度手术（在浅层音频层上截断视频梯度，保留跨模态交互层的梯度）；按区域重新加权损失</td><td>LTX-2</td></tr>
<tr><td><a href="https://arxiv.org/html/2609.29816">AV-GRPO</a></td><td>多模态奖励纠缠，信用分配不清；同时优化两个塔成本高</td><td>以一个模态为锚做 rollout；冻结一个塔、只训另一个塔（trajectory-locked frozen-tower）；按模态调整目标和扰动强度；提出 5DAV 解耦数据集</td><td>LTX-2.3（LoRA 和全参数都有结果）</td></tr>
<tr><td><a href="https://arxiv.org/html/2605.12179v1">SyncDPO</a></td><td>MSE 对时间错位不敏感</td><td>规则构造时间扭曲负样本，按课程学习加大难度</td><td>—</td></tr>
</table>
<div class="sv-box sv-warn">
<p><strong>对 H3 的启示</strong>：H3 是单流 DiT，视频和音频 token 在同一序列里，并没有 OmniNFT / AV-GRPO 假设的“双塔”结构，所以逐层梯度手术、冻结一个塔这类做法不能直接照搬。但<strong>按模态分配优势值或权重</strong>的思想可以用：视频质量奖励只作用于视频潜变量的损失，同步奖励同时作用于两者，音质奖励只作用于音频潜变量的损失。</p>
</div>
</details>
</div>

### 3.6 奖励模型

<div class="sv-card">
<h4>VideoReward、SyncReward、VA-Judger 等</h4>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：奖励模型给每个视频打一个分。训练它时<strong>不要求分数的绝对值准确，只要求“人觉得 A 比 B 好时，A 的分更高”</strong>。</p>
<table>
<tr><th>量</th><th>大白话</th></tr>
<tr><td>分差</td><td>A 的分 − B 的分</td></tr>
<tr><td>胜率</td><td>模型认为“A 比 B 好”的概率，由分差决定</td></tr>
<tr><td>平局</td><td>人标“差不多”；保留这类标注能让奖励模型更稳</td></tr>
</table>
<div class="sv-oneline">P(A 胜 B) = σ(A 的分 − B 的分)；训练：人选 A 时让这个概率变大</div>
<p><strong>数字例子</strong>：A 得 2.0 分，B 得 1.0 分，P(A 胜) = σ(1) ≈ 0.73。如果人也选了 A，损失 = −log 0.73 ≈ 0.31；如果人选了 B，损失 = −log 0.27 ≈ 1.31，模型会把 B 的分往上调、A 的往下调。</p>
<p><strong>VideoReward 的两个关键经验</strong>：</p>
<ul>
<li>保留“平局”标注（GSB 里的 S），而不是丢掉；</li>
<li>画质、运动、文本对齐<strong>每个维度单独输出一个分</strong>，不要混成一个总分。</li>
</ul>
<p><strong>注意</strong>：veRL-Omni 默认的 CLAP + ImageBind 只看语义是否对得上，不看画质，也不看参考图像不像。</p>
</div>
<details class="sv-deep">
<summary>详细版：Bradley-Terry 与带平局的 BTT、各模型对比（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>s<sub>A</sub>, s<sub>B</sub></td><td>奖励模型给 A、B 的分</td></tr>
<tr><td>σ</td><td>sigmoid</td></tr>
<tr><td>θ<sub>tie</sub></td><td>平局参数（θ<sub>tie</sub> &gt; 1，越大越容易判平局）</td></tr>
</table>
<p><strong>Bradley-Terry（BT）</strong>：P(A ≻ B) = σ(s<sub>A</sub> − s<sub>B</sub>) = e<sup>s<sub>A</sub></sup> / (e<sup>s<sub>A</sub></sup> + e<sup>s<sub>B</sub></sup>)；损失 = −log P(人选的那一方胜)。</p>
<p><strong>带平局的 BTT</strong>（Rao-Kupper 形式，VideoAlign 采用这一类扩展）：</p>
<div class="sv-formula">
P(A ≻ B) = e<sup>s<sub>A</sub></sup> / (e<sup>s<sub>A</sub></sup> + θ<sub>tie</sub>·e<sup>s<sub>B</sub></sup>)，　P(A ≈ B) = (θ<sub>tie</sub>² − 1)·e<sup>s<sub>A</sub></sup>e<sup>s<sub>B</sub></sup> / [ (e<sup>s<sub>A</sub></sup> + θ<sub>tie</sub>e<sup>s<sub>B</sub></sup>)(e<sup>s<sub>B</sub></sup> + θ<sub>tie</sub>e<sup>s<sub>A</sub></sup>) ]
</div>
<p>三种结果（A 胜、B 胜、平局）的概率之和为 1；损失为人工标注结果的负对数概率。分差越接近 0，平局概率越大。</p>
<table>
<tr><th>模型</th><th>数据规模</th><th>评估维度</th><th>关键设计</th></tr>
<tr><td><a href="https://github.com/KlingAIResearch/VideoAlign">VideoReward</a>（VideoAlign）</td><td>18.2 万条人工偏好，覆盖 12 个视频模型，1.6 万个 prompt</td><td>画质（VQ）、运动质量（MQ）、文本对齐（TA）</td><td>Bradley-Terry 比分数回归更好（数据越多差距越小）；<strong>保留平局标注（BTT）</strong>显著提升鲁棒性；<strong>每个维度使用独立 token</strong>，避免与上下文无关的维度（VQ、MQ）被 prompt 耦合</td></tr>
<tr><td><a href="https://arxiv.org/html/2610.16338">SyncReward</a></td><td>8 万条专业同步评分，覆盖 8 个音视频生成器</td><td>音视频同步（语音、起始音、乐器、环境声四类）</td><td>在真实片段上做语义-时间对比预训练，再在生成片段上用人工监督微调；在 Best-of-16 和 DiffusionNFT 后训练中都领先其他同步指标；报告发现 Synchformer、AV-Align 这类专项指标与人类判断差距明显</td></tr>
<tr><td><a href="https://sharelab-sii.github.io/VA-Judger/">VA-Judger</a></td><td>VAPref-10K：9K 个 prompt，1.03 万对细粒度比较</td><td>文本、视频、音频三者的整体语义和时间连贯性</td><td>思维链式全模态奖励模型；指出用分项指标拼接奖励容易被作弊</td></tr>
<tr><td><a href="https://arxiv.org/html/2406.15252">VideoScore</a></td><td>3.76 万个视频的多维评分</td><td>多维质量</td><td>较早期的工作，VideoAlign 测得它在现代模型上泛化差</td></tr>
<tr><td>CLAP + ImageBind</td><td>—</td><td>文本-音频、音频-视频语义对齐</td><td>veRL-Omni 的 H3 默认奖励；<strong>不衡量参考图一致性，也不衡量画质</strong></td></tr>
</table>
</details>
</div>

### 3.7 推理时提升

<div class="sv-card">
<h4>Best-of-N 与 Flow-NRG</h4>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：不改模型，生成时多花算力。<strong>Best-of-N</strong> 是生成 N 个挑最好的；<strong>Flow-NRG</strong> 是在每一步去噪时，顺着“奖励变高”的方向推一下。</p>
<div class="sv-oneline">Best-of-N：输出 = 奖励最高的那个；　Flow-NRG：每一步的方向 = 模型方向 + 推力 × 奖励升高的方向</div>
<p><strong>数字例子</strong>：生成 4 个视频，奖励 6.1、7.3、5.2、6.8，输出 7.3 那个，成本是 4 倍（H3 约 4 × 35 = 140 GPU·分钟）。</p>
<p><strong>最重要的用途</strong>：Best-of-4 是训练方案<strong>至少要超过的基线</strong>。DPO 后单次生成如果还不如它，训练就白做了。</p>
</div>
<details class="sv-deep">
<summary>详细版（点击展开）</summary>
<table class="sv-symtab">
<tr><th>本卡符号</th><th>含义</th></tr>
<tr><td>R</td><td>奖励模型（Flow-NRG 需要能处理带噪 latent 的版本）</td></tr>
<tr><td>λ</td><td>引导强度</td></tr>
<tr><td>∇<sub>x<sub>t</sub></sub>R</td><td>奖励对当前带噪状态的梯度</td></tr>
</table>
<p><strong>示意形式</strong>（通用奖励引导写法，不是 Flow-NRG 原文公式）：v'(x<sub>t</sub>, t) = v<sub>θ</sub>(x<sub>t</sub>, t) − λ·∇<sub>x<sub>t</sub></sub>R(x<sub>t</sub>, t)。负号来自 v = ε − x<sub>0</sub> 的约定（采样时 x 沿 −v 方向走）；多目标时 R = Σ w<sub>k</sub>R<sub>k</sub>。</p>
<ul>
<li><strong>Best-of-N</strong>：生成 N 个结果，用奖励模型挑最好的。零训练、零风险，推理成本变为 N 倍。SyncReward 的 Best-of-16 实验显示，选用的奖励模型决定了提升幅度：用同步专项指标选出的结果，在专项指标上分数最高，但人类偏好分反而低。</li>
<li><strong>Flow-NRG</strong>：在含噪视频上直接施加奖励梯度引导，可以给多个目标分配不同权重。需要一个能处理含噪潜变量的奖励模型。</li>
<li><strong>用途</strong>：Best-of-N 的结果是训练方案<strong>至少要超过的基线</strong>。如果 DPO 后的单次生成还不如基础模型的 Best-of-4，训练就没有意义。</li>
</ul>
</details>
</div>

### 3.8 奖励作弊与运动坍缩

<div class="sv-card">
<h4>CoRe：奖励模型和生成器共同演化</h4>
<p class="sv-src"><a href="https://arxiv.org/html/2609.36245">CoRe</a>, 2026</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：奖励模型是个“考官”，生成器训练久了会<strong>摸清考官的盲区</strong>，专门生成考官给高分、但人看着很差的视频。CoRe 的办法是<strong>考官也跟着一起更新</strong>，同时用真实视频的偏好钉住标准。</p>
<div class="sv-oneline">循环：生成器按奖励训练几步 → 用生成器的最新样本 + 真实视频偏好重新训练奖励模型 → 继续</div>
<p><strong>数字例子</strong>（论文观察）：奖励模型固定不变时，奖励分一直在涨，画面却第 15 步开始变糊，第 24 步出现网格伪影，第 100 步彻底坍缩。运动幅度指标从约 85 掉到约 18：<strong>视频几乎不动了</strong>，但“平滑度”“一致性”这些指标反而更高。</p>
<p><strong>教训</strong>：只看奖励曲线是不够的，必须同时监控运动幅度，并定期人工抽检。</p>
</div>
<details class="sv-deep">
<summary>详细版（点击展开）</summary>
<ul>
<li><strong>现象</strong>：针对固定的潜空间奖励模型优化，几百次更新内生成器就会跑出奖励模型的训练分布（distributional escape）。奖励分数依然很高，画质和运动却在恶化：约第 15 步开始模糊，第 24 步出现网格伪影，第 100 步彻底坍缩。</li>
<li><strong>静止视频陷阱</strong>：没有锚定的在线奖励模型短暂提升后坍缩，dynamic degree 从峰值 83.72–89.67 掉到 16.67–19.44，同时“主体一致性、平滑度”反而上升，因为静止画面在这些指标上天然得高分。</li>
<li><strong>做法</strong>：持续用生成器当前的样本重新训练奖励模型，同时用真实视频的偏好做锚定，使生成器无法靠偏离数据分布来拿高分。</li>
</ul>
</details>
</div>
<div class="sv-card">
<h4>GraftOOD：混入外部参考样本</h4>
<p class="sv-src"><a href="https://arxiv.org/html/2610.13470">GraftOOD</a>, 2026</p>
<div class="sv-simple">
<span class="sv-label">简洁版</span>
<p><strong>一句话</strong>：在线训练时，每组里<strong>掺进一两个外部的好视频</strong>（比如真实视频或更强模型的输出），让模型除了“和自己比”，还能看到“外面的好样本长什么样”，从而不会越训越不动。</p>
<div class="sv-oneline">一组 = 若干个自己生成的视频 + 少量外部参考视频，一起算组内相对好坏，一起做速度回归</div>
<p><strong>数字例子</strong>（论文，Wan2.1-1.3B）：运动幅度从 4.86 升到 44.41；代价是 HPSv3 美学分从 8.10 降到 7.32。</p>
<p><strong>对 H3</strong>：每个 prompt 都已经有一条 Seedance 2.5 视频，可以直接当外部参考样本。前提是确认这批数据可以合规地用于训练。</p>
</div>
<details class="sv-deep">
<summary>详细版（点击展开）</summary>
<ul>
<li>即使使用视频级的运动质量奖励，奖励优化也会出现运动坍缩。</li>
<li>做法：在部分 prompt 组里，用外部视频样本（分布外的参考样本）替换一部分在线 rollout，以共享的速度回归目标同时学习“组内相对奖励”和“参考样本”。</li>
<li>Wan2.1-1.3B 上，dynamic degree 从 4.86 升到 44.41（对比只用奖励的 FlowAWR），代价是 HPSv3 从 8.10 降到 7.32。</li>
<li><strong>对 H3</strong>：每个 prompt 都已经有一条 Seedance 2.5 视频，天然就是“格式匹配的外部参考样本”。可以用来防止运动坍缩，但要先确认 Seedance 输出用于训练的合规性。</li>
</ul>
</details>
</div>

## 4. 业界实践

<table>
<tr><th>模型</th><th>后训练流程</th><th>偏好或 RL 阶段的具体做法</th><th>来源</th></tr>
<tr><td>Seedance 1.0（字节）</td><td>SFT → 视频专用 RLHF</td><td>3 个奖励模型（基础、运动、美学），直接最大化加权奖励；奖励模型和扩散模型多轮迭代；偏好按维度逐个标注，并要求胜出方在其他维度上不能更差；RLHF 也用在蒸馏后的超分模型上</td><td><a href="https://arxiv.org/html/2506.09113v2">技术报告</a></td></tr>
<tr><td>HunyuanVideo 1.5（腾讯，8.3B）</td><td>CT → SFT → RLHF，文生视频和图生视频分开训练</td><td><strong>文生视频</strong>：约 1 万个均衡 prompt，SFT 模型每个 prompt 生成 N 个候选，人工按 <strong>GSB</strong> 在语义保真度、运动质量、美学三个维度上标注，先做 DPO，再做在线 RL。<strong>图生视频</strong>：直接用 VLM 奖励模型（文本对齐、图像对齐、画质、运动四个维度）加 MixGRPO 在线 RL。采样时变换 seed 和 CFG scale 以增加多样性</td><td><a href="https://huangrh99.github.io/obsidian-papers/image-generation-posttrain/2025-11-HunyuanVideo">报告解读</a></td></tr>
<tr><td>Kling-Omni（快手）</td><td>—</td><td>人工标注运动和画面完整性的偏好对，做 DPO；因成本放弃 GRPO；步数从 150 蒸馏到 10</td><td>转引自<a href="https://shreshthsaini.github.io/blogs/rl-for-video.html">综述博客</a></td></tr>
<tr><td>LongCat-Video（美团）</td><td>—</td><td>3 个奖励模型，组大小 G=4；在 16 步中只在 0–6 步内随机选一步做 SDE；共 500 次迭代</td><td>同上</td></tr>
<tr><td>Cosmos-Predict2.5（NVIDIA）</td><td>—</td><td>VideoAlign 奖励，每组 8 个 rollout × 20 步，用扩散损失作为正则项</td><td>同上</td></tr>
<tr><td>Wan 2.1 / 2.2（阿里）</td><td>在更高质量的数据上继续训练</td><td>报告未披露奖励模型或 RLHF 阶段</td><td><a href="https://cflabs.ai/research/video/where-open-video-models-fall-short/">CFLabs</a></td></tr>
</table>
<h4>从业界实践提炼的经验</h4>
<ol>
<li><strong>先离线、后在线</strong>：先用人工偏好做 DPO 打好起点，再用奖励模型做在线 RL。原因是奖励模型难以捕捉细粒度的运动质量，直接做 RL 容易被作弊。</li>
<li><strong>分维度标注，并加“不能更差”的约束</strong>：只标一个总体胜负，噪声大且信号纠缠。分维度标注后，可以分别构造偏好对，也可以加权合成总体偏好。</li>
<li><strong>候选来自当前最强的检查点</strong>：偏好对应来自待训练模型本身（on-policy），而不是其他模型。</li>
<li><strong>大厂也在控制 RL 成本</strong>：Kling 因成本放弃 GRPO，LongCat 只在一步上做 SDE，Cosmos 只用 20 步采样。对 33B 的 H3 来说，成本约束只会更紧。</li>
<li><strong>奖励模型要迭代</strong>：Seedance 和 CoRe 都表明，固定奖励模型会被生成器“甩开”。</li>
</ol>

## 5. 方法对比与选型

<table>
<tr><th>方法</th><th>需要人工标注</th><th>需要奖励模型</th><th>每次更新的采样成本</th><th>额外显存</th><th>工程量（H3）</th><th>主要风险</th><th>建议</th></tr>
<tr><td>Flow-DPO（离线）</td><td>是（GSB）</td><td>否</td><td>无（提前生成）</td><td>无（LoRA 关闭即参考模型）</td><td>中：需要在现有 LoRA 训练框架上改损失</td><td>数据过时；β 过小导致过拟合</td><td><span class="sv-tag sv-t-hi">首选</span></td></tr>
<tr><td>SyncDPO 式自动负样本</td><td>否</td><td>否</td><td>无</td><td>无</td><td>低</td><td>只学到避开人工扭曲</td><td><span class="sv-tag sv-t-hi">与 Flow-DPO 混合</span></td></tr>
<tr><td>LocalDPO</td><td>否</td><td>否</td><td>每个 prompt 推理一次</td><td>无</td><td>中：需要掩码补全</td><td>同上</td><td><span class="sv-tag sv-t-mid">可选</span></td></tr>
<tr><td>DiffusionNFT（在线）</td><td>否</td><td>是</td><td>每个 prompt 4–8 次，可用 10 步采样</td><td>旧策略 LoRA 副本（很小）</td><td>低到中：veRL-Omni 已有配方</td><td>奖励作弊；运动坍缩</td><td><span class="sv-tag sv-t-hi">第二阶段</span></td></tr>
<tr><td>DiffusionART</td><td>否</td><td>是（可以是成对模型）</td><td>较少</td><td>小</td><td>高：没有 H3 实现</td><td>新方法，未经大规模验证</td><td><span class="sv-tag sv-t-mid">后续升级</span></td></tr>
<tr><td>Flow-GRPO / DanceGRPO</td><td>否</td><td>是</td><td>每个 prompt 4–8 次 + SDE 轨迹</td><td>轨迹或计算图</td><td>高：Ref2VA 没有 SDE rollout</td><td>成本失控</td><td><span class="sv-tag sv-t-lo">不推荐</span></td></tr>
<tr><td>奖励直接反传</td><td>否</td><td>是（必须可求导）</td><td>每步一次，需要反传过 VAE</td><td>很高</td><td>高</td><td>显存、奖励作弊</td><td><span class="sv-tag sv-t-lo">不推荐</span></td></tr>
<tr><td>全参数 SFT / DPO</td><td>—</td><td>—</td><td>—</td><td>约 530 GB 优化器状态，需要 16 卡以上或 CPU offload</td><td>高</td><td>灾难性遗忘</td><td><span class="sv-tag sv-t-lo">不推荐</span></td></tr>
</table>

## 6. H3 现成训练工具

<p>MiniMax 官方只提供推理，没有训练代码。社区整理在 <a href="https://github.com/MiniMax-AI/awesome-minimax-h3-integration">awesome-minimax-h3-integration</a>。下面列出与我们方案直接相关的几个。</p>
<div class="sv-card">
<h4>MiniMax-H3-FineTuning（监督 LoRA 训练）<span class="sv-tag sv-t-hi">阶段 2 底座</span></h4>
<p class="sv-src"><a href="https://github.com/IAmIronMan42/MiniMax-H3-FineTuning">IAmIronMan42/MiniMax-H3-FineTuning</a></p>
<ul>
<li>基于官方 Diffusers 集成的 rectified flow 训练器，带潜变量缓存预处理，<strong>音频也计入损失</strong>。</li>
<li>可训练参数有三种模式：<code>heads</code>（只训练 <code>proj_out</code> 和 <code>audio_proj_out</code>，约 100 万参数，单卡冒烟测试用）；<code>lora</code>（PEFT rank 16，作用于 <code>to_qkv</code>、<code>to_out.0</code>、<code>linear_1</code>、<code>linear_2</code>，约 1070 万参数）；<code>all</code>（全部 33B，需要 DeepSpeed ZeRO-3）。</li>
<li>作者报告在 8×A800 上用 2000 条约 30 秒的视频片段（448×768，每条序列约 6.5 万 token，包含真实立体声）完成了 LoRA 训练。</li>
<li>文档 <code>FIXES.md</code> 记录了 9 个坑：5 个是 H3 特有的数值约定（处理不当会悄无声息地损坏权重），4 个是 LoRA、ZeRO-3 和长序列组合下的问题。<strong>改造前务必通读。</strong></li>
<li>没有 DPO 损失，需要自己加。好在缓存好的 latent、文本编码和 LoRA 模块都可以直接复用。</li>
</ul>
<pre><code># 官方 README 中的流程（节选）
python prepare_cache.py --metadata train.jsonl --output cache/train \
  --model /path/to/MiniMax-H3 --height 256 --width 256 --frames 22 \
  --encode-text --encode-audio
deepspeed --num_gpus 8 train.py --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/lora --max-steps 800 \
  --trainable lora --strategy deepspeed</code></pre>
</div>
<div class="sv-card">
<h4>veRL-Omni（在线 DiffusionNFT）<span class="sv-tag sv-t-hi">阶段 4 底座</span></h4>
<p class="sv-src"><a href="https://verl-omni.readthedocs.io/en/latest/examples/minimax_h3/diffusionnft_trainer_minimax_h3.html">MiniMax H3 T2VA, FL2VA, and Ref2VA DiffusionNFT</a>（文档更新于 2026-09-23）</p>
<ul>
<li>训练 rank-64 的 H3 LoRA：Actor 用 FSDP2 训练 Diffusers transformer，rollout 由 vLLM-Omni 负责生成视频和音频。</li>
<li><strong>Ref2VA 支持</strong>：每行最多 12 个参考文件（最多 9 张图、3 段视频、3 段音频）；参考素材行保持固定，只对生成的视频和音频加噪、训练。数据格式为 JSONL（<code>prompt</code>、<code>images</code>、<code>videos</code>、<code>audios</code>），用 <code>prepare_ref2va_data.py</code> 转成 parquet。<strong>和我们现有的 jsonl 很接近</strong>。</li>
<li><strong>检查点</strong>：rollout 用 <code>Ref2VA/</code>，Actor 用 <code>transformer_ref/</code>。我们之前考虑删掉的 diffusers 格式“重复”目录，<strong>在这里是必需的</strong>，不能删。</li>
<li><strong>采样配置</strong>：训练 rollout 默认 10 步（<code>INFER_STEPS=10</code>），验证用 40 步；FL2VA 训练分辨率 288×448，验证 576×928；Ref2VA 训练参考图短边 512，验证 1024。</li>
<li><strong>并行与显存</strong>：默认 rollout TP=4，开启 CPU offload。文档说明 TP=2 时，在 96 GB 卡上同步权重时显存不够。H200 有 141 GB，可能会宽松一些，需要实测。8 卡配方 <code>ROLLOUT_N=4</code>。<code>ACTOR_SP</code> 可开 Ulysses，但必须能整除 56 个注意力头。</li>
<li><strong>默认奖励</strong>：CLAP 加 ImageBind（音视频语义对齐）。<strong>文档明确说明这两个奖励不衡量参考图相似度</strong>，评估参考一致性需要另外接入参考感知的奖励模型。</li>
<li>默认 Attention 后端为 FlashAttention 3（需要 <code>kernels</code> 包）；rollout 不开 FP8 量化（已知有问题）。</li>
</ul>
<pre><code>MODEL_PATH=&lt;MiniMax-H3-root&gt; \
DATA_DIR=&lt;parquet-directory&gt; \
REF_IMAGE_SHORT_EDGE=512 \
VAL_REF_IMAGE_SHORT_EDGE=1024 \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_ref2va_lora.sh</code></pre>
</div>
<div class="sv-card">
<h4>其他</h4>
<ul>
<li><a href="https://github.com/X-GenGroup/Flow-Factory/blob/main/examples/grpo/lora/minimax_h3_fl2va/default.yaml">Flow-Factory</a>：H3 的 GRPO LoRA 配方（rank 32，组大小 8，24 步，512×768，124 帧，batch=1，<code>guidance_scale=1.0</code>）。作者注明尚未在完整检查点上验证显存、画质和数值一致性。</li>
<li><a href="https://github.com/Tencent-Hunyuan/UniRL/issues/587">UniRL</a>（腾讯混元）：H3 只支持 t2va 训练侧，LoRA-only（Flow-GRPO rank 32，DiffusionNFT rank 64）。issue 中估算全参数训练约 530 GB 状态，并建议用 <code>text_embed_cache_path</code> 缓存文本编码，避免加载 32B 编码器。</li>
<li>VeOmni（字节，有 FL2VA、Ref2VA 配置）、musubi-tuner、diffusion-pipe、SimpleTuner：都有不同程度的 H3 LoRA 支持，可作为备选。</li>
</ul>
</div>

## 7. 落地方案

<div class="sv-phase" id="p0">

### 阶段 0：评测基线（约 1 周）

<p><strong>目标</strong>：在动训练之前，先确定“怎样才算变好”。</p>
<ol>
<li><strong>评测集</strong>：从 2095 条中分层抽样 200–300 条，<strong>永不参与训练</strong>。分层维度包括：是否有对白（口型）、单主体还是多主体、参考图数量、是否有参考音频、时长（4–6 秒、7–10 秒、11–15 秒）、场景类型。</li>
<li><strong>基线生成</strong>：基础模型、固定 seed，用现有的 <code>infer_jsonl.py</code> 生成。</li>
<li><strong>自动指标</strong>（每次评测都跑同一套）：
<ul>
<li>画质、运动、文本对齐：VideoReward 的三个维度分数；</li>
<li>音视频同步：SyncReward（若可获取）或 Synchformer DeSync；</li>
<li>运动幅度：光流平均幅度或 VBench dynamic degree，<strong>用来监控运动坍缩</strong>；</li>
<li>参考一致性：生成帧与参考图的人脸或主体特征相似度（例如 ArcFace、DINOv2）；</li>
<li>音频：CLAP 文本-音频相似度。</li>
</ul>
</li>
<li><strong>Best-of-4 基线</strong>：每条用 4 个 seed 生成，按 VideoReward 总分选最优，与单次生成做一次 GSB，得出“只靠推理就能拿到的提升”。</li>
</ol>
<p><strong>交付物</strong>：评测集 jsonl、基线视频、指标表、Best-of-4 的 GSB 结果。</p>
</div>
<div class="sv-phase" id="p1">

### 阶段 1：偏好数据生成与 GSB 标注（2–4 周，主要瓶颈是生成算力）

<h4>1.1 候选生成</h4>
<ul>
<li><strong>同一条件、多个 seed</strong>：同一组 prompt 和参考素材，生成 K=4 个 seed 的结果。不改步数、flow_shift 和条件噪声，确保差异只来自随机性。</li>
<li><strong>规模</strong>：第一轮 2000–3000 个 prompt（HunyuanVideo 1.5 用了约 1 万）。现有 1744 条可直接运行的样本可以全部用上；prompt 不够时，再用 LLM 改写出变体来扩充。</li>
<li><strong>分辨率和步数</strong>：标注用的视频要和最终使用的设置一致（768p、50 步）。训练时可以把 latent 降采样缓存，但标注时看到的必须是真实效果。</li>
<li><strong>Seedance 2.5 视频的用法</strong>：<strong>不建议</strong>直接作为 DPO 的胜出方。它不是 H3 的输出分布，构成的偏好对差异过大，模型会学到“模仿 Seedance 风格”，而不是修正自己的缺陷；同时它的潜变量在 H3 下的似然本来就低，DPO 信号会被这部分差异主导。如果用，建议作为阶段 4 的外部参考样本（GraftOOD 思路）或者少量 SFT 数据，并<strong>先确认服务条款是否允许</strong>。</li>
</ul>
<h4>1.2 GSB 标注规范</h4>
<table>
<tr><th>维度</th><th>判断要点</th></tr>
<tr><td>画面质量</td><td>清晰度、伪影、崩坏（手、脸、文字）、闪烁</td></tr>
<tr><td>运动质量</td><td>运动是否自然、物理是否合理、<strong>动作幅度是否充分</strong>（不能因为静止而判好）</td></tr>
<tr><td>音视频同步</td><td>口型与对白对齐、音效与画面事件对齐</td></tr>
<tr><td>音频质量</td><td>人声清晰度、音色与参考音频一致、背景音乐和音效是否合理</td></tr>
<tr><td>参考一致性</td><td>人物身份、服装、物体与参考图一致</td></tr>
<tr><td>指令遵循</td><td>镜头、动作、对白内容是否按 prompt 执行</td></tr>
</table>
<ul>
<li>每个维度都给出 G / S / B（A 好 / 持平 / B 好），另加一个整体判断。</li>
<li><strong>采纳规则</strong>（参考 Seedance）：整体 G（或 B），并且胜出方在<strong>任何维度上都不差于</strong>落败方，才进入 DPO 训练集。其余配对保留下来，留给阶段 3 训练奖励模型（平局同样有用）。</li>
<li><strong>质量控制</strong>：约 10% 的配对由两人重复标注，统计一致率；混入已知答案的金标题；A/B 左右位置随机。</li>
<li><strong>配对数量</strong>：4 个 seed 可以组成 6 对，全部标注成本较高。可以先每个 prompt 标 2–3 对，或者先用奖励模型预排序，只标最高分和最低分、次高和次低。</li>
</ul>
<h4>1.3 自动构造的同步负样本（SyncDPO 思路，几乎零成本）</h4>
<ul>
<li>取人工判定为“好”的样本，把音频潜变量在时间轴上平移 ±2–8 个潜变量帧，或截取片段错位拼接，得到“同步更差”的负样本，与原样本组成偏好对。</li>
<li>按课程学习从大偏移逐步过渡到小偏移。混合比例建议不超过总偏好对的 20–30%，避免模型只学会“避开人为错位”。</li>
</ul>
<p><strong>交付物</strong>：<code>preference_pairs.jsonl</code>，每行包含条件路径、胜出和落败的视频路径、各维度标签、标注人和一致性标记；以及给奖励模型用的全量标注表。</p>
</div>
<div class="sv-phase" id="p2">

### 阶段 2：LoRA Flow-DPO（2–3 周）

<h4>2.1 训练框架</h4>
<ul>
<li>基于 MiniMax-H3-FineTuning 改造：沿用它的 latent 缓存、文本编码缓存、LoRA 注入和 ZeRO-3 支持，<strong>只替换损失函数</strong>。</li>
<li>每个样本同时加载胜出和落败的视频及音频 latent，<strong>共享同一个 t 和同一份噪声 ε</strong>（这是 Diffusion-DPO 的标准做法，可以降低方差）。</li>
<li><strong>参考模型</strong>：对同一批输入，在禁用 LoRA 的状态下前向一次，得到 v<sub>ref</sub>（使用 <code>torch.no_grad</code>）。这样每个样本需要 4 次前向（胜出和落败各一次当前模型、一次参考模型），但不需要额外的权重显存。</li>
</ul>
<h4>2.2 损失（伪代码）</h4>
<pre><code>t   = sample_timestep()              # 与 H3 预训练使用的分布 / shift 一致
eps = randn_like(x_w)
x_t_w = (1 - t) * x_w + t * eps ; v_w = eps - x_w   # 按 H3 实际的 flow 约定调整符号
x_t_l = (1 - t) * x_l + t * eps ; v_l = eps - x_l
&nbsp;
err = lambda v_pred, v: w_vid * mse(v_pred.video, v.video) + w_aud * mse(v_pred.audio, v.audio)
&nbsp;
with lora_enabled():  e_w, e_l = err(model(x_t_w, c, t), v_w), err(model(x_t_l, c, t), v_l)
with lora_disabled(), no_grad(): r_w, r_l = err(model(x_t_w, c, t), v_w), err(model(x_t_l, c, t), v_l)
&nbsp;
logits = -0.5 * beta * ((e_w - r_w) - (e_l - r_l))   # β 为常数，不乘 (1-t)^2
loss   = -logsigmoid(logits).mean() + lambda_sft * e_w.mean()   # 可选：对胜出样本加一项 SFT 正则</code></pre>
<p>说明：最后的 SFT 正则项是 LLM DPO 中常见的稳定手段，用来防止胜出样本的似然也一起下降。是否需要，取决于训练时 <code>e_w</code> 是否持续上升。flow 的方向约定（t=0 是数据还是噪声、v 的符号）<strong>必须和 H3 的实现一致</strong>，这是 FineTuning 仓库 <code>FIXES.md</code> 里强调的坑之一。</p>
<h4>2.3 起始超参（建议值，需要扫参）</h4>
<table>
<tr><th>参数</th><th>起点</th><th>说明</th></tr>
<tr><td>LoRA rank / alpha</td><td>32 / 64（可以试 64 / 128）</td><td>社区 RL 配方用 32–64；FineTuning 仓库做 SFT 用 16</td></tr>
<tr><td>LoRA 目标层</td><td><code>to_qkv</code>、<code>to_out.0</code>、<code>linear_1</code>、<code>linear_2</code></td><td>可另做一个对比实验：只训 AdaLN 分支</td></tr>
<tr><td>β（常数）</td><td>扫描若干个数量级</td><td>β 越大越保守。训练时盯住隐式奖励准确率和 margin，准确率很快到 100% 往往意味着过拟合</td></tr>
<tr><td>学习率</td><td>1e-5 – 5e-5（LoRA）</td><td>UniRL 的 RL 配方为 1e-4，<code>adam_epsilon=1e-12</code>（LoRA 梯度很小）</td></tr>
<tr><td>步数</td><td>先跑 300–1000 步，每 100–200 步评测一次</td><td>DPO 收敛快，主要防止过训练</td></tr>
<tr><td>音视频权重</td><td>w_vid : w_aud 按 token 数归一化后约 1 : 1</td><td>做一次消融</td></tr>
<tr><td>训练分辨率</td><td>先用低分辨率 latent（如 448p）跑通，再上原分辨率</td><td>注意 17n+5 帧约束</td></tr>
</table>
<h4>2.4 验收</h4>
<ul>
<li>评测集上与基础模型做 GSB：<strong>净胜率显著为正</strong>，并且超过阶段 0 的 Best-of-4。</li>
<li>自动指标：运动幅度<strong>不得明显下降</strong>；参考一致性不下降；同步指标提升。</li>
<li>回归检查：未参与训练的场景类型（如纯风景、无对白）没有明显退化。</li>
<li>部署：LoRA 通过 SGLang 的 <code>--lora-path</code> 加载（你们的 infra 已经打了 <code>patch_sglang_h3_lora.py</code>），可以用 <code>lora_scale</code> 调节强度做插值评测。</li>
</ul>
<p><strong>迭代</strong>：用 DPO 后的模型重新生成候选、重新标注，再做第二轮 DPO（iterative DPO），可以缓解离线数据过时的问题。一般两到三轮之后收益递减。</p>
</div>
<div class="sv-phase" id="p3">

### 阶段 3：用标注训练奖励模型（2–3 周，可与阶段 2 并行）

<ul>
<li><strong>架构</strong>：在支持视频和音频输入的多模态模型（如 Qwen 系列全模态模型）上微调。参考 VideoReward：<strong>每个维度一个独立的打分 token</strong>，损失用<strong>带平局的 Bradley-Terry（BTT）</strong>。</li>
<li><strong>数据</strong>：阶段 1 的全部配对（包括平局和维度冲突的配对，它们对奖励模型有用），加上阶段 2 迭代时的新标注。按 prompt 划分训练集和验证集。</li>
<li><strong>同步维度</strong>：优先评估能否直接用 SyncReward；如果和我们的标注一致率高，就不需要自己训练这个维度。</li>
<li><strong>验收</strong>：在留出的配对上，各维度的成对准确率接近标注员之间的一致率。另外检查奖励模型能否识别“静止视频”，可以人为构造静止样本来测试，避免它偏爱静止画面。</li>
<li><strong>用途</strong>：(1) Best-of-N 选优；(2) 预排序以减少人工标注；(3) 阶段 4 的在线奖励。</li>
</ul>
</div>
<div class="sv-phase" id="p4">

### 阶段 4：在线 DiffusionNFT（可选，3–4 周）

<ul>
<li><strong>起点</strong>：阶段 2 的 DPO LoRA 合并后作为新的基础模型，或者在其上继续训练 LoRA。</li>
<li><strong>框架</strong>：veRL-Omni Ref2VA 配方，rank 64；把现有 jsonl 转成它的格式（<code>prompt</code>、<code>images</code>、<code>audios</code>，注意音频需要是 wav 或 mp3，并满足时长约束）。</li>
<li><strong>奖励组合</strong>：阶段 3 的奖励模型（画质、运动、指令遵循、参考一致性）+ 同步奖励 + 运动幅度奖励（RVM 的经验：没有运动奖励就会趋向静止）+ CLAP / ImageBind。按模态加权：画面类奖励只作用于视频损失，音质类只作用于音频损失（参考 OmniNFT）。</li>
<li><strong>采样</strong>：训练 rollout 10–16 步、低分辨率；每 N 步用 40–50 步全分辨率做验证。</li>
<li><strong>防作弊</strong>：(1) 混入真实或外部参考样本作为锚定（GraftOOD 的思路，可以用 Seedance 2.5 视频，前提是合规）；(2) 每隔几百步用新样本补充标注、更新奖励模型（CoRe 的思路）；(3) 固定频率人工抽检，<strong>只要奖励在涨而人工 GSB 不涨，立即停止训练</strong>。</li>
<li><strong>后续升级</strong>：如果 DiffusionNFT 不稳定或成本过高，可以换成 DiffusionART（支持成对奖励，报告 GPU 时间少约 12 倍），需要自己移植到 H3。</li>
</ul>
</div>

## 8. 算力与时间估算

<div class="sv-box sv-warn"><p>以下是根据本地实测推算的<strong>粗略估计</strong>，没有考虑 8 卡 Ulysses 的加速比、失败重试和排队时间，实际部署前需要用小批量实测校准。</p></div>
<h4>8.1 候选生成（阶段 1，最大头）</h4>
<p>基准：768p、50 步、2×H200 TP2，平均约 1063 秒/条，约等于 <strong>35 GPU·分钟/条</strong>。</p>
<table>
<tr><th>配置</th><th>视频数</th><th>GPU·小时</th><th>64 卡（8 个 8 卡节点）墙钟时间</th></tr>
<tr><td>2000 prompt × 4 seed，50 步</td><td>8,000</td><td>约 4,700</td><td>约 3 天</td></tr>
<tr><td>3000 prompt × 4 seed，50 步</td><td>12,000</td><td>约 7,000</td><td>约 4.6 天</td></tr>
<tr><td>2000 prompt × 4 seed，30 步（需先确认画质可接受）</td><td>8,000</td><td>约 2,900</td><td>约 1.9 天</td></tr>
<tr><td>只用现有 1 个 2 卡 pod</td><td>8,000</td><td>约 4,700</td><td><strong>约 98 天</strong>（不可行）</td></tr>
</table>
<p>结论：<strong>阶段 1 必须上集群</strong>。参考现有的 k8s manifest，用多个 8 卡 pod 并行起服务，再把 <code>infer_jsonl.py</code> 改成多端点分发。8 卡 Ulysses8 的单条耗时需要先实测。</p>
<h4>8.2 DPO 训练（阶段 2）</h4>
<ul>
<li>每个偏好对需要 4 次 transformer 前向（其中 2 次带梯度）。以 5 秒 768p 视频为例，序列长度在数万 token 量级，单节点 8 卡 ZeRO-3 加梯度检查点即可运行。</li>
<li>数千个偏好对、约 1000 步，预计 1–3 天/轮（单个 8 卡节点），具体取决于分辨率和时长。</li>
</ul>
<h4>8.3 总体时间线</h4>
<table>
<tr><th>周</th><th>工作</th></tr>
<tr><td>1</td><td>阶段 0：评测集、基线、Best-of-4；同时搭建多节点生成服务</td></tr>
<tr><td>2–4</td><td>阶段 1：候选生成，标注规范试标（先 100 对校准一致率），然后正式标注；同时改造 DPO 训练代码，用小数据跑通</td></tr>
<tr><td>4–6</td><td>阶段 2：第一轮 DPO、扫参、评测；阶段 3：开始训练奖励模型</td></tr>
<tr><td>6–8</td><td>第二轮迭代 DPO；奖励模型验收</td></tr>
<tr><td>8–12</td><td>阶段 4（可选）：DiffusionNFT</td></tr>
</table>

## 9. 风险与应对

<table>
<tr><th>风险</th><th>征兆</th><th>应对</th></tr>
<tr><td>运动坍缩（视频变静态）</td><td>运动幅度指标下降，一致性指标反常上升</td><td>运动单列为标注维度，胜出方运动不得更差；加运动奖励；混入外部参考样本</td></tr>
<tr><td>奖励作弊</td><td>奖励持续上涨，人工 GSB 不涨或下降</td><td>奖励模型迭代更新；真实数据锚定；固定频率人工抽检，设置早停</td></tr>
<tr><td>DPO 过拟合</td><td>隐式奖励准确率很快到 100%，胜出样本误差 <code>e_w</code> 也上升</td><td>增大 β；减少步数；加 SFT 正则；扩充数据</td></tr>
<tr><td>离线数据过时</td><td>第二轮评测收益明显变小</td><td>迭代 DPO，或转在线训练</td></tr>
<tr><td>音视频互相拖累</td><td>画质提升而音质或同步变差，或反过来</td><td>分模态加权损失；在分维度标注中分别检查</td></tr>
<tr><td>参考一致性下降</td><td>人物身份漂移</td><td>参考一致性作为必检维度；加入特征相似度奖励</td></tr>
<tr><td>flow 约定或数值错误</td><td>训练初期损失异常，或 LoRA 为 0 时输出就和基础模型不一致</td><td>通读 FineTuning 仓库 <code>FIXES.md</code>；先做“LoRA 全零时输出与 SGLang 推理一致”的对齐测试</td></tr>
<tr><td>标注噪声大</td><td>重复标注一致率低于 70%</td><td>细化规范；对高分歧配对做多人投票；只保留明确胜负的配对</td></tr>
<tr><td>合规</td><td>—</td><td>用 Seedance 输出训练前确认服务条款；veRL-Omni 公开数据集中的 FLUX 生成图是非商业许可</td></tr>
</table>

## 10. 不建议的做法

<div class="sv-box sv-bad">
<ul>
<li><strong>全参数训练</strong>：约 530 GB 优化器状态，需要 16 卡以上或 CPU offload；容易遗忘；社区没有验证过的配方。</li>
<li><strong>在 50 步完整采样上做 GRPO</strong>：每组 4–8 个 rollout，一次迭代就要数小时；Ref2VA 也没有现成的 SDE rollout。</li>
<li><strong>直接用原版 Diffusion-DPO 的时间步加权</strong>：在 flow 模型上会导致奖励作弊和伪影。</li>
<li><strong>只用自动构造的负样本，不做人工标注</strong>：模型只学会避开人工制造的瑕疵，学不到真实的质量偏好。</li>
<li><strong>用 Seedance 2.5 视频当胜出方、H3 输出当落败方做 DPO</strong>：分布差异过大，学到的是风格模仿，而不是缺陷修正。</li>
<li><strong>用改步数、改条件噪声的方式制造偏好对</strong>：模型会学到“步数差异”这种捷径信号。</li>
<li><strong>只看自动指标就发布</strong>：所有工作都显示，专项指标与人类判断可能背离，最终验收必须用人工 GSB。</li>
</ul>
</div>

## 附录 A：Flow-DPO 逐步推导

<div class="sv-simple">
<span class="sv-label">简洁版：推导只有四句话</span>
<ol>
<li><strong>想要的</strong>：模型得分越高越好，但不能离原模型太远。</li>
<li><strong>数学结论</strong>：满足这个要求的最优模型，它对某个视频的“偏爱程度”（相对原模型）就等于这个视频的奖励。所以<strong>不用单独训练奖励模型</strong>，直接让“好视频的偏爱程度 &gt; 差视频的偏爱程度”就行，这就是 DPO。</li>
<li><strong>扩散模型的麻烦</strong>：“偏爱程度”算不出来，但可以用“去噪误差比原模型小多少”来近似。误差越小，越偏爱。</li>
<li><strong>换成 flow 写法</strong>：误差从“噪声误差”换算成“方向误差”，会多出一个随噪声变化的系数 (1−t)²。实践中这个系数让高噪声段几乎没有约束，于是改成常数。</li>
</ol>
<div class="sv-oneline">最终：得分 = 好视频的进步量 − 差视频的进步量，损失 = −log σ(β/2 × 得分)（和 Flow-DPO 卡片的简洁版一致）</div>
<p>下面的 11 步是这四句话的严格展开，可以只在需要核对时再看。</p>
</div>
<p>推导分三段：</p>
<ol>
<li><strong>从 RLHF 到 DPO</strong>（第 1–4 步）：把“训练奖励模型再做 RL”化简为直接在偏好对上做监督学习。</li>
<li><strong>从 DPO 到 Diffusion-DPO</strong>（第 5–8 步）：扩散模型的样本似然不可计算，需要换成可计算的逐步去噪误差。</li>
<li><strong>从 Diffusion-DPO 到 Flow-DPO</strong>（第 9–11 步）：把噪声预测改写为速度预测，得到 β<sub>t</sub> = β(1−t)²，再说明为什么改用常数 β。</li>
</ol>
<h4>第 0 步：符号</h4>
<table>
<tr><th>符号</th><th>含义</th></tr>
<tr><td>c</td><td>条件（prompt、参考图、参考音频）</td></tr>
<tr><td>x<sub>0</sub></td><td>干净样本（视频和音频 latent）；x<sub>0</sub><sup>w</sup> 是胜出样本，x<sub>0</sub><sup>l</sup> 是落败样本</td></tr>
<tr><td>p<sub>θ</sub></td><td>正在训练的模型（策略）</td></tr>
<tr><td>p<sub>ref</sub></td><td>参考模型（冻结的基础模型；LoRA 场景下就是关掉 adapter 的模型）</td></tr>
<tr><td>r(c, x)</td><td>人类偏好对应的（未知）奖励函数</td></tr>
<tr><td>β</td><td>KL 正则系数</td></tr>
<tr><td>σ(·)</td><td>sigmoid 函数，σ(z) = 1 / (1 + e<sup>−z</sup>)</td></tr>
<tr><td>q</td><td>扩散的前向（加噪）过程，是已知的高斯分布</td></tr>
</table>

### A.1 从 RLHF 到 DPO

<h4>第 1 步：带 KL 约束的奖励最大化目标</h4>
<p>RLHF 希望模型生成高奖励样本，同时不要离参考模型太远，以免作弊或崩坏：</p>
<div class="sv-formula">
max<sub>p<sub>θ</sub></sub>　𝔼<sub>c, x∼p<sub>θ</sub>(x|c)</sub> [ r(c, x) ] − β · KL( p<sub>θ</sub>(x|c) ‖ p<sub>ref</sub>(x|c) )
</div>
<p>β 就在这里登场：它是“追奖励”和“保持像原模型”之间的权衡系数。</p>
<h4>第 2 步：求出最优策略的闭式解</h4>
<p>把 KL 展开，目标写成（对每个 c 单独看）：</p>
<div class="sv-formula">
𝔼<sub>x∼p</sub> [ r(x) − β log p(x) + β log p<sub>ref</sub>(x) ]
= −β · 𝔼<sub>x∼p</sub> [ log p(x) − log ( p<sub>ref</sub>(x) · e<sup>r(x)/β</sup> ) ]
</div>
<p>引入归一化常数 Z(c) = Σ<sub>x</sub> p<sub>ref</sub>(x|c) · e<sup>r(c,x)/β</sup>，定义分布 p*(x|c) = p<sub>ref</sub>(x|c) · e<sup>r(c,x)/β</sup> / Z(c)，则上式等于：</p>
<div class="sv-formula">
−β · KL( p ‖ p* ) + β · log Z(c)
</div>
<p>log Z(c) 与 p 无关，KL ≥ 0 并且当且仅当 p = p* 时取 0，所以最优策略就是：</p>
<div class="sv-formula">
p*(x|c) = (1 / Z(c)) · p<sub>ref</sub>(x|c) · exp( r(c, x) / β )
</div>
<h4>第 3 步：反解出奖励</h4>
<p>对上式两边取对数并整理：</p>
<div class="sv-formula">
r(c, x) = β · log [ p*(x|c) / p<sub>ref</sub>(x|c) ] + β · log Z(c)
</div>
<p>这一步的意义是：<strong>奖励可以完全用“最优策略与参考策略的对数似然比”来表示</strong>，不再需要单独的奖励模型。</p>
<h4>第 4 步：代入 Bradley-Terry 偏好模型，得到 DPO</h4>
<p>人类偏好用 Bradley-Terry 模型描述，即 x<sup>w</sup> 胜过 x<sup>l</sup> 的概率由两者奖励之差决定：</p>
<div class="sv-formula">
P( x<sup>w</sup> ≻ x<sup>l</sup> | c ) = σ( r(c, x<sup>w</sup>) − r(c, x<sup>l</sup>) )
</div>
<p>把第 3 步代入，<strong>β·log Z(c) 在相减时抵消</strong>（同一个 c）。再用待训练的 p<sub>θ</sub> 替代 p*，对偏好数据做最大似然，就得到 DPO 损失：</p>
<div class="sv-formula">
L<sub>DPO</sub>(θ) = −𝔼<sub>(c, x<sup>w</sup>, x<sup>l</sup>)</sub> [ log σ( β · log [ p<sub>θ</sub>(x<sup>w</sup>|c) / p<sub>ref</sub>(x<sup>w</sup>|c) ] − β · log [ p<sub>θ</sub>(x<sup>l</sup>|c) / p<sub>ref</sub>(x<sup>l</sup>|c) ] ) ]
</div>
<p>直观含义：让模型相对参考模型<strong>提高</strong>胜出样本的似然、<strong>降低</strong>落败样本的似然；β 控制这种调整的力度。</p>

### A.2 从 DPO 到 Diffusion-DPO

<h4>第 5 步：扩散模型的困难在于 p<sub>θ</sub>(x<sub>0</sub>|c) 不可计算</h4>
<p>扩散模型通过多步去噪生成样本，x<sub>0</sub> 的边缘似然需要对所有去噪路径积分：</p>
<div class="sv-formula">
p<sub>θ</sub>(x<sub>0</sub>|c) = ∫ p<sub>θ</sub>(x<sub>0:T</sub>|c) dx<sub>1:T</sub>，　p<sub>θ</sub>(x<sub>0:T</sub>) = p(x<sub>T</sub>) · Π<sub>t=1</sub><sup>T</sup> p<sub>θ</sub>(x<sub>t−1</sub> | x<sub>t</sub>)
</div>
<p>这个积分算不出来，所以 DPO 损失不能直接用。Diffusion-DPO（Wallace et al.）的做法是把奖励和 KL 都定义在<strong>整条路径 x<sub>0:T</sub></strong> 上，从第 1 步重新走一遍，得到路径级的 DPO：</p>
<div class="sv-formula">
L = −𝔼 [ log σ( β · 𝔼<sub>x<sub>1:T</sub>∼p<sub>θ</sub>(x<sub>1:T</sub>|x<sub>0</sub><sup>w</sup>)</sub> [ log p<sub>θ</sub>(x<sub>0:T</sub><sup>w</sup>) / p<sub>ref</sub>(x<sub>0:T</sub><sup>w</sup>) ] − β · 𝔼<sub>x<sub>1:T</sub>∼p<sub>θ</sub>(x<sub>1:T</sub>|x<sub>0</sub><sup>l</sup>)</sub> [ log p<sub>θ</sub>(x<sub>0:T</sub><sup>l</sup>) / p<sub>ref</sub>(x<sub>0:T</sub><sup>l</sup>) ] ) ]
</div>
<h4>第 6 步：用前向过程 q 近似反向后验，并拆成逐步的形式</h4>
<p>p<sub>θ</sub>(x<sub>1:T</sub>|x<sub>0</sub>) 依然难以采样，于是用已知的前向加噪过程 q(x<sub>1:T</sub>|x<sub>0</sub>) 来近似。由于路径对数似然比是 T 个单步项之和，这个和可以写成“均匀采样一个 t，再乘以 T”：</p>
<div class="sv-formula">
𝔼<sub>q</sub> [ log p<sub>θ</sub>(x<sub>0:T</sub>) / p<sub>ref</sub>(x<sub>0:T</sub>) ] = T · 𝔼<sub>t∼U(1,T)</sub> 𝔼<sub>q(x<sub>t</sub>|x<sub>0</sub>)</sub> [ log p<sub>θ</sub>(x<sub>t−1</sub>|x<sub>t</sub>) / p<sub>ref</sub>(x<sub>t−1</sub>|x<sub>t</sub>) ]
</div>
<p>又因为 −log σ(·) 是凸函数，根据 Jensen 不等式可以把对 t 和 x<sub>t</sub> 的期望移到 −log σ 外面，得到原损失的一个<strong>上界</strong>，优化这个上界即可。每一步的单步对数似然比再用 KL 散度表达（对 q(x<sub>t−1</sub>|x<sub>t</sub>, x<sub>0</sub>) 取期望），得到：</p>
<div class="sv-formula">
L ≤ −𝔼<sub>t, x<sub>t</sub><sup>w</sup>, x<sub>t</sub><sup>l</sup></sub> log σ( −βT · [ KL(q<sup>w</sup> ‖ p<sub>θ</sub><sup>w</sup>) − KL(q<sup>w</sup> ‖ p<sub>ref</sub><sup>w</sup>) − ( KL(q<sup>l</sup> ‖ p<sub>θ</sub><sup>l</sup>) − KL(q<sup>l</sup> ‖ p<sub>ref</sub><sup>l</sup>) ) ] )
</div>
<p>其中 q<sup>w</sup> = q(x<sub>t−1</sub><sup>w</sup> | x<sub>t</sub><sup>w</sup>, x<sub>0</sub><sup>w</sup>)，p<sub>θ</sub><sup>w</sup> = p<sub>θ</sub>(x<sub>t−1</sub><sup>w</sup> | x<sub>t</sub><sup>w</sup>)，其余同理。注意出现了<strong>负号</strong>：KL 越小，说明模型越能解释这个样本，似然越高。</p>
<h4>第 7 步：高斯 KL 化为噪声预测误差</h4>
<p>q(x<sub>t−1</sub>|x<sub>t</sub>, x<sub>0</sub>) 和 p<sub>θ</sub>(x<sub>t−1</sub>|x<sub>t</sub>) 都是方差相同（σ<sub>t</sub>²I）的高斯分布，因此：</p>
<div class="sv-formula">
KL( q ‖ p<sub>θ</sub> ) = ‖ μ̃<sub>t</sub>(x<sub>t</sub>, x<sub>0</sub>) − μ<sub>θ</sub>(x<sub>t</sub>, t) ‖² / (2σ<sub>t</sub>²)
</div>
<p>在 ε 参数化下，均值之差与噪声预测之差成正比：μ̃<sub>t</sub> − μ<sub>θ</sub> = k<sub>t</sub> · (ε<sub>θ</sub>(x<sub>t</sub>, t) − ε)，k<sub>t</sub> 只依赖噪声调度。于是：</p>
<div class="sv-formula">
KL( q ‖ p<sub>θ</sub> ) = ω<sub>t</sub> · ‖ ε − ε<sub>θ</sub>(x<sub>t</sub>, t) ‖²，　ω<sub>t</sub> = k<sub>t</sub>² / (2σ<sub>t</sub>²)
</div>
<p>参考模型的 KL 有完全相同的形式。如果 KL 里有与模型无关的常数项，它们在“θ 减 ref”时也会抵消。</p>
<h4>第 8 步：得到 Diffusion-DPO 损失</h4>
<p>代回第 6 步，并把 T·ω<sub>t</sub> 等常数并入系数（原论文实践中把 ω<sub>t</sub> 取为常数，Flow-DPO 论文统一写成 β/2）：</p>
<div class="sv-formula">
L<sub>DD</sub>(θ) = −𝔼 [ log σ( −(β/2) · ( ‖ε<sup>w</sup> − ε<sub>θ</sub>(x<sub>t</sub><sup>w</sup>, t)‖² − ‖ε<sup>w</sup> − ε<sub>ref</sub>(x<sub>t</sub><sup>w</sup>, t)‖² − ( ‖ε<sup>l</sup> − ε<sub>θ</sub>(x<sub>t</sub><sup>l</sup>, t)‖² − ‖ε<sup>l</sup> − ε<sub>ref</sub>(x<sub>t</sub><sup>l</sup>, t)‖² ) ) ) ]
</div>
<p>至此，“似然比”被替换成了“去噪误差之差”：<strong>模型在胜出样本上的误差比参考模型下降得越多、在落败样本上的误差比参考模型上升得越多，sigmoid 内的值就越大，损失越小</strong>。所有量都只需要一次前向就能算出。</p>

### A.3 从 Diffusion-DPO 到 Flow-DPO

<h4>第 9 步：rectified flow 的参数化</h4>
<p>rectified flow 在数据和噪声之间做直线插值，模型预测速度 v 而不是噪声 ε（Flow-DPO 论文的约定：t=0 是数据，t=1 是噪声）：</p>
<div class="sv-formula">
x<sub>t</sub> = (1 − t) · x<sub>0</sub> + t · ε，　v = dx<sub>t</sub>/dt = ε − x<sub>0</sub>
</div>
<h4>第 10 步：把 ε 误差换算成 v 误差</h4>
<p>(a) 由 x<sub>t</sub> 和 v 反推 ε。从 v = ε − x<sub>0</sub> 得 x<sub>0</sub> = ε − v，代入 x<sub>t</sub> 的定义：</p>
<div class="sv-formula">
x<sub>t</sub> = (1 − t)(ε − v) + t·ε = ε − (1 − t)·v　⇒　ε = x<sub>t</sub> + (1 − t)·v
</div>
<p>(b) 模型预测 v<sub>θ</sub> 时，对应的隐式噪声预测是 ε<sub>θ</sub> = x<sub>t</sub> + (1 − t)·v<sub>θ</sub>（同一个 x<sub>t</sub>）。</p>
<p>(c) 两式相减，x<sub>t</sub> 抵消：</p>
<div class="sv-formula">
ε − ε<sub>θ</sub> = (1 − t) · (v − v<sub>θ</sub>)　⇒　‖ε − ε<sub>θ</sub>‖² = (1 − t)² · ‖v − v<sub>θ</sub>‖²
</div>
<p>参考模型同理：‖ε − ε<sub>ref</sub>‖² = (1 − t)² · ‖v − v<sub>ref</sub>‖²。</p>
<h4>第 11 步：代入得到 Flow-DPO，并改用常数 β</h4>
<p>把第 10 步代入第 8 步，四个误差项都带同一个因子 (1 − t)²，可以提到括号外，与 β 合并：</p>
<div class="sv-formula">
L<sub>FD</sub>(θ) = −𝔼 [ log σ( −(β<sub>t</sub>/2) · ( ‖v<sup>w</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>w</sup>, t)‖² − ‖v<sup>w</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>w</sup>, t)‖² − ( ‖v<sup>l</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>l</sup>, t)‖² − ‖v<sup>l</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>l</sup>, t)‖² ) ) ) ]，　β<sub>t</sub> = β(1 − t)²
</div>
<p><strong>这就是 β<sub>t</sub> = β(1−t)² 的来历</strong>：它不是额外设计的，而是从 ε 参数化严格换算过来的。它在 t→1（高噪声）时趋于 0，KL 约束在高噪声段几乎失效，模型会在这些决定构图和运动的步上过拟合偏好，导致奖励作弊和伪影。</p>
<p>Flow-DPO 借鉴了 DDPM 的经验（DDPM 训练时丢掉理论推导出的时间步权重，样本质量反而更好），<strong>把 β<sub>t</sub> 直接换成常数 β</strong>。这样做在理论上相当于换了一个时间步加权方式，实验上在所有维度都更稳定。最终实际使用的形式是：</p>
<div class="sv-formula">
L<sub>FD</sub>(θ) = −𝔼 [ log σ( −(β/2) · ( Δ<sup>w</sup> − Δ<sup>l</sup> ) ) ]，　Δ<sup>*</sup> = ‖v<sup>*</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>*</sup>, t)‖² − ‖v<sup>*</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>*</sup>, t)‖²
</div>
<h4>与阶段 2 伪代码的对应关系</h4>
<table>
<tr><th>公式中的量</th><th>伪代码变量</th></tr>
<tr><td>‖v<sup>w</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>w</sup>, t)‖²</td><td><code>e_w</code>（开启 LoRA）</td></tr>
<tr><td>‖v<sup>w</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>w</sup>, t)‖²</td><td><code>r_w</code>（关闭 LoRA，<code>no_grad</code>）</td></tr>
<tr><td>‖v<sup>l</sup> − v<sub>θ</sub>(x<sub>t</sub><sup>l</sup>, t)‖² / ‖v<sup>l</sup> − v<sub>ref</sub>(x<sub>t</sub><sup>l</sup>, t)‖²</td><td><code>e_l</code> / <code>r_l</code></td></tr>
<tr><td>Δ<sup>w</sup> − Δ<sup>l</sup></td><td><code>(e_w - r_w) - (e_l - r_l)</code></td></tr>
<tr><td>−(β/2)·(…)</td><td><code>logits = -0.5 * beta * (...)</code></td></tr>
<tr><td>H3 的音视频联合</td><td>‖·‖² 换成 <code>w_vid · mse(video) + w_aud · mse(audio)</code></td></tr>
</table>
<div class="sv-box sv-warn">
<p><strong>推导中的近似</strong>（实际实现时值得知道）：</p>
<ol>
<li>第 4 步用 p<sub>θ</sub> 替代 p*，是 DPO 本身的做法；</li>
<li>第 6 步用前向过程 q 替代反向后验，并用 Jensen 不等式得到上界，优化的是上界而不是原目标；</li>
<li>第 8 步把时间步权重 ω<sub>t</sub> 并入常数；</li>
<li>第 11 步把 β<sub>t</sub> 换成常数 β，这是经验选择，不是推导结果。</li>
</ol>
<p>另外，理论上 x<sub>t</sub><sup>w</sup> 和 x<sub>t</sub><sup>l</sup> 的噪声可以独立采样；实践中通常共享同一个 t 和同一份 ε 以降低方差。</p>
</div>

## 11. 参考文献

<h4>偏好优化</h4>
<ol class="sv-refs">
<li>Liu et al. Improving Video Generation with Human Feedback (Flow-DPO / VideoReward). NeurIPS 2025. <a href="https://arxiv.org/abs/2501.13918">arXiv:2501.13918</a> · <a href="https://github.com/KlingAIResearch/VideoAlign">code</a></li>
<li>Huang et al. Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models (LocalDPO). CVPR 2026. <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Mind_the_Generative_Details_Direct_Localized_Detail_Preference_Optimization_for_CVPR_2026_paper.pdf">PDF</a></li>
<li>Li et al. DynamicsBoost: Annotation-Free Continuation Preference Optimization. CVPR 2026. <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Li_DynamicsBoost_Dynamic_Plausible_Video_Generation_via_Annotation-Free_Continuation_Preference_Optimization_CVPR_2026_paper.pdf">PDF</a></li>
<li>SyncDPO: Enhancing Temporal Synchronization in Video-Audio Joint Generation via Preference Learning. 2026. <a href="https://arxiv.org/html/2605.12179v1">arXiv:2605.12179</a></li>
<li>OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization. WACV 2026. <a href="https://arxiv.org/html/2412.15159v1">arXiv:2412.15159</a></li>
<li>Diffusion-APO: Trajectory-Aware Direct Preference Alignment for Video Diffusion Transformers. 2026. <a href="https://arxiv.org/html/2605.07503v1">arXiv:2605.07503</a></li>
</ol>
<h4>强化学习</h4>
<ol class="sv-refs" start="7">
<li>Zheng et al. DiffusionNFT: Online Diffusion Reinforcement with Forward Process. ICLR 2026. <a href="https://arxiv.org/html/2509.16117">arXiv:2509.16117</a> · <a href="https://github.com/NVlabs/DiffusionNFT">code</a></li>
<li>Scaling Reinforcement Learning for Diffusion Models via Velocity Matching (RVM). 2026. <a href="https://arxiv.org/abs/2608.23664">arXiv:2608.23664</a></li>
<li>DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models. 2026. <a href="https://arxiv.org/abs/2610.12572">arXiv:2610.12572</a></li>
<li>Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization. 2026. <a href="https://arxiv.org/html/2605.15980">arXiv:2605.15980</a></li>
<li>LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL. 2026. <a href="https://arxiv.org/pdf/2609.03528.pdf">arXiv:2609.03528</a></li>
<li>Advances in GRPO for Generation Models: A Survey. 2026. <a href="https://arxiv.org/abs/2603.06623">arXiv:2603.06623</a></li>
<li>Video Generation Models: A Survey of Post-Training and Alignment. 2026. <a href="https://arxiv.org/html/2610.00812">arXiv:2610.00812</a></li>
</ol>
<h4>音视频联合生成</h4>
<ol class="sv-refs" start="14">
<li>OmniNFT: Modality-wise Omni Diffusion Reinforcement for Joint Audio-Video Generation. 2026. <a href="https://arxiv.org/html/2605.12480">arXiv:2605.12480</a></li>
<li>AV-GRPO: Modality-Anchored Decoupling Diffusion RL for Joint Audio-Video Generation. 2026. <a href="https://arxiv.org/html/2609.29816">arXiv:2609.29816</a></li>
<li>SyncReward: Human-Aligned Synchronization Reward for Audio-Visual Generation. 2026. <a href="https://arxiv.org/html/2610.16338">arXiv:2610.16338</a></li>
<li>VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation. <a href="https://sharelab-sii.github.io/VA-Judger/">project page</a></li>
</ol>
<h4>奖励作弊</h4>
<ol class="sv-refs" start="18">
<li>CoRe: Co-Evolving Reward Models for Mitigating Latent Reward Hacking in Video Diffusion Models. 2026. <a href="https://arxiv.org/html/2609.36245">arXiv:2609.36245</a></li>
<li>GraftOOD: Grafting Out-of-Distribution Exemplars into Advantage-Weighted Rectification for Video Reward Optimization. 2026. <a href="https://arxiv.org/html/2610.13470">arXiv:2610.13470</a></li>
<li>VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation. <a href="https://arxiv.org/html/2406.15252">arXiv:2406.15252</a></li>
</ol>
<h4>业界报告与解读</h4>
<ol class="sv-refs" start="21">
<li>Seedance 1.0: Exploring the Boundaries of Video Generation Models. <a href="https://arxiv.org/html/2506.09113v2">arXiv:2506.09113</a></li>
<li>HunyuanVideo 1.5 技术报告解读. <a href="https://huangrh99.github.io/obsidian-papers/image-generation-posttrain/2025-11-HunyuanVideo">Paper Trail</a></li>
<li>RL for Video: The State of the Space. <a href="https://shreshthsaini.github.io/blogs/rl-for-video.html">blog</a></li>
<li>Where open video models actually fall short. <a href="https://cflabs.ai/research/video/where-open-video-models-fall-short/">CFLabs</a></li>
</ol>
<h4>H3 工具</h4>
<ol class="sv-refs" start="25">
<li>MiniMax-H3-FineTuning. <a href="https://github.com/IAmIronMan42/MiniMax-H3-FineTuning">GitHub</a></li>
<li>veRL-Omni: MiniMax H3 DiffusionNFT recipes. <a href="https://verl-omni.readthedocs.io/en/latest/examples/minimax_h3/diffusionnft_trainer_minimax_h3.html">docs</a></li>
<li>Flow-Factory: H3 FL2VA GRPO LoRA config. <a href="https://github.com/X-GenGroup/Flow-Factory/blob/main/examples/grpo/lora/minimax_h3_fl2va/default.yaml">GitHub</a></li>
<li>UniRL issue #587: full-parameter RL for MiniMax-H3. <a href="https://github.com/Tencent-Hunyuan/UniRL/issues/587">GitHub</a></li>
<li>awesome-minimax-h3-integration. <a href="https://github.com/MiniMax-AI/awesome-minimax-h3-integration">GitHub</a></li>
</ol>
<p class="sv-meta" style="margin-top:40px">文档位置：<code>MiniMax-H3/H3_posttraining_survey.html</code>，单文件、无外部依赖，可离线打开。</p>
