# 视频生成学习站（2026）

基于《视频生成技术学习手册（2026）》整理的章节式学习网站，并补充了数据管线专题（镜头切分、视频 / 图像去重、头部团队数据管线对比）和 MiniMax-H3 后训练专题（第 4 章 4.A，GSB → Flow-DPO → 在线 RL 的调研与落地方案）。

## 运行

纯静态站点，无需构建。浏览器不允许 `file://` 页面读取本地 Markdown，所以需要起一个本地服务：

```bash
cd VideoGenLearn
./serve.sh          # 默认端口 8765，可传参：./serve.sh 9000
```

脚本会启动 `python3 -m http.server` 并自动打开 http://localhost:8765 ，按 Ctrl+C 停止。

## 目录结构

```
VideoGenLearn/
├── serve.sh              # 本地预览脚本
├── index.html            # 页面外壳
├── H3_posttraining_survey.html  # H3 后训练专题的原文（单页 HTML），4.A 由它生成
├── tools/
│   └── import_survey.py  # 把上面的原文转成 content/04a-h3-posttraining.md
├── assets/
│   ├── chapters.js       # 章节清单（顺序、编号、标题、文件）
│   ├── app.js            # 路由、Markdown 渲染、公式、搜索、目录
│   └── style.css         # 样式（支持明暗主题、移动端）
└── content/              # 每章一个 Markdown 文件
    ├── 00-overview.md
    ├── 01-data.md
    ├── 01a-shot-segmentation.md
    ├── 01b-dedup.md
    ├── 01c-industry-pipelines.md
    ├── 02-architecture.md
    ├── 03-midtraining.md
    ├── 04-posttraining.md
    ├── 04a-h3-posttraining.md   # 4.A：H3 后训练专题（自动生成，不要手改）
    ├── 05-reference.md
    ├── 06-training-efficiency.md
    ├── 07-inference.md
    └── 08-roadmap.md
```

## 新增或修改章节

1. 在 `content/` 下新建或编辑 `.md` 文件。
2. 新文件需要在 `assets/chapters.js` 中登记；带 `parent` 字段的条目会显示为该章的子专题。

**H3 后训练专题（4.A）**：内容来自 `H3_posttraining_survey.html`，修改时改原文，再重新生成：

```bash
python3 tools/import_survey.py
```

脚本把 h2/h3 转成 Markdown 标题（目录和搜索都能用），其余保留原 HTML（class 加 `sv-` 前缀，样式在 `assets/style.css` 末尾），原文里的站内锚点改写成 `#/页面id/sN`。原文新增或删除 h2 时，要同步改脚本里 `PAGES` 的 h2 列表。

## Markdown 扩展语法

- 公式：行内 `$...$`，独立行 `$$...$$`（KaTeX 渲染）。
- 提示框：`> [!结论]`、`> [!提示]`、`> [!注意]`、`> [!经验]`、`> [!来源]`。
- 流程图：
  ````
  ```flow
  步骤一 | 说明文字
  步骤二 | 说明文字
  ```
  ````
- 站内链接：`[文字](#/章节id)`，章节 id 见 `chapters.js`。

## 功能

- 左侧章节导航、右侧本页目录（随滚动高亮）
- 全文搜索（按 `/` 聚焦，回车跳转第一个结果）
- 明暗主题切换、移动端适配

依赖通过 CDN 加载：marked 12.0.2、KaTeX 0.16.9。
