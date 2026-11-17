# 视频生成学习站（2026）

基于《视频生成技术学习手册（2026）》整理的章节式学习网站，并补充了数据管线专题（镜头切分、视频 / 图像去重、头部团队数据管线对比）。

## 运行

纯静态站点，无需构建。浏览器不允许 `file://` 页面读取本地 Markdown，所以需要起一个本地服务：

```bash
cd VideoGenLearn
python3 -m http.server 8765
```

然后打开 http://localhost:8765 。

## 目录结构

```
VideoGenLearn/
├── index.html            # 页面外壳
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
    ├── 05-reference.md
    ├── 06-training-efficiency.md
    ├── 07-inference.md
    └── 08-roadmap.md
```

## 新增或修改章节

1. 在 `content/` 下新建或编辑 `.md` 文件。
2. 新文件需要在 `assets/chapters.js` 中登记；带 `parent` 字段的条目会显示为该章的子专题。

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
