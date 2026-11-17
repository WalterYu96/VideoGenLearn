// 章节清单：新增章节只需在 content/ 下加 .md 文件并在这里登记
window.CHAPTERS = [
  { id: 'overview', num: '0', title: '导读：2026 年的格局', file: 'content/00-overview.md', stage: '总览' },
  { id: 'data', num: '1', title: '数据', file: 'content/01-data.md', stage: '能力' },
  { id: 'data-shot', num: '1.A', title: '专题：镜头切分', file: 'content/01a-shot-segmentation.md', parent: 'data' },
  { id: 'data-dedup', num: '1.B', title: '专题：视频与图像去重', file: 'content/01b-dedup.md', parent: 'data' },
  { id: 'data-industry', num: '1.C', title: '专题：顶级团队视频数据管线', file: 'content/01c-industry-pipelines.md', parent: 'data' },
  { id: 'architecture', num: '2', title: '预训练：模型架构', file: 'content/02-architecture.md', stage: '能力' },
  { id: 'midtraining', num: '3', title: '中训练（Continue Training）', file: 'content/03-midtraining.md', stage: '能力' },
  { id: 'posttraining', num: '4', title: '后训练：SFT、RL、蒸馏、OPD', file: 'content/04-posttraining.md', stage: '对齐与成本' },
  { id: 'reference', num: '5', title: '参考生成与多任务能力', file: 'content/05-reference.md', stage: '能力' },
  { id: 'train-eff', num: '6', title: '训练效率', file: 'content/06-training-efficiency.md', stage: '对齐与成本' },
  { id: 'inference', num: '7', title: '推理效率', file: 'content/07-inference.md', stage: '对齐与成本' },
  { id: 'roadmap', num: '8', title: '学习路线与参考资料', file: 'content/08-roadmap.md', stage: '总览' },
];
