#!/usr/bin/env python3
"""把 H3_posttraining_survey.html 转成学习站第 4 章下的专题页 content/04a-h3-posttraining.md。

做法：h2 / h3 转成 Markdown 标题（这样右侧目录和全文搜索都能用），其余部分保留原 HTML，
class 加上 sv- 前缀（避免和站点自己的 .card / .flow 冲突，样式在 assets/style.css 的 sv- 一节）。
站内锚点 #xxx 改写成 #/<页面id>/s<N>，N 与 app.js 的 assignIds 编号规则一致（按 h2、h3 顺序）。

用法（原文更新后重跑即可覆盖生成）：
    python3 tools/import_survey.py [H3_posttraining_survey.html]
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "H3_posttraining_survey.html"

# 页面 id、文件、标题、包含的 h2 id（目前整篇合成一页；要拆页就在这里加条目）
PAGES = [
    ("h3-rl", "04a-h3-posttraining.md", "MiniMax-H3 后训练调研与落地方案",
     ["tldr", "context", "landscape", "industry", "compare", "ecosystem", "plan", "compute", "risks", "notdo",
      "appendix-dpo", "refs"]),
]
TITLE = "MiniMax-H3 后训练调研与落地方案"


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def main() -> None:
    doc = SRC.read_text(encoding="utf-8")
    body = re.search(r"<main>(.*)</main>", doc, re.S).group(1)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    meta = strip_tags(re.search(r'<p class="meta">(.*?)</p>', body, re.S).group(1))
    intro = body[: body.index("<h2")]
    intro = re.sub(r"<h1>.*?</h1>", "", intro, flags=re.S)
    intro = re.sub(r'<p class="meta">.*?</p>', "", intro, flags=re.S)

    # 按 h2 切段
    parts = [p for p in re.split(r'(?=<h2 id=")', body[body.index("<h2"):]) if p.strip()]
    sections = {re.match(r'<h2 id="([^"]+)"', p).group(1): p for p in parts}
    assigned = [s for _, _, _, ids in PAGES for s in ids]
    missing = set(sections) - set(assigned)
    if missing:
        sys.exit(f"有 h2 没分配到页面：{missing}")

    # 第一遍：算出每个锚点落在哪页的第几个标题
    anchors: dict[str, str] = {}
    page_html = {}
    for pid, _, _, ids in PAGES:
        chunk = (intro if pid == PAGES[0][0] else "") + "".join(sections[i] for i in ids)
        page_html[pid] = chunk
        n = 0
        pending = []  # 非标题元素的 id（如 div.phase#p1）指向它后面的第一个标题
        for m in re.finditer(r'<(h2|h3)\b[^>]*>|<(?!h[23]\b)[a-z0-9]+\b[^>]*\bid="([^"]+)"', chunk):
            if m.group(1):
                n += 1
                hid = re.search(r'id="([^"]+)"', m.group(0))
                if hid:
                    anchors[hid.group(1)] = f"#/{pid}/s{n}"
                for a in pending:
                    anchors[a] = f"#/{pid}/s{n}"
                pending = []
            else:
                pending.append(m.group(2))

    for pid, fname, title, _ in PAGES:
        out = convert(page_html[pid], anchors)
        head = [f"# {title}", ""]
        head += ["> [!来源] 原文", f"> {meta}。由 `tools/import_survey.py` 从 `H3_posttraining_survey.html` 生成。", ""]
        (ROOT / "content" / fname).write_text("\n".join(head) + "\n" + out.strip() + "\n", encoding="utf-8")
        print(f"写入 content/{fname}")


def convert(chunk: str, anchors: dict[str, str]) -> str:
    # 1. 先把 <pre> 整块保护起来（保留缩进；空行换成 &nbsp; 行，避免 marked 在空行处结束 HTML 块）
    pres = []

    def keep_pre(m):
        pres.append(re.sub(r"\n[ \t]*\n", "\n&nbsp;\n", m.group(0)))
        return f"{len(pres) - 1}"

    chunk = re.sub(r"<pre\b.*?</pre>", keep_pre, chunk, flags=re.S)

    # 2. 站内锚点
    def fix_href(m):
        target = anchors.get(m.group(1))
        if not target:
            sys.exit(f"找不到锚点 #{m.group(1)}")
        return f'href="{target}"'

    chunk = re.sub(r'href="#([^"/][^"]*)"', fix_href, chunk)

    # 3. 外链在站内统一由 app.js 加 target，去掉原有的
    # 4. class 加前缀
    chunk = re.sub(r'class="([^"]+)"', lambda m: 'class="' + " ".join("sv-" + c for c in m.group(1).split()) + '"', chunk)

    # 5. h2 / h3 → Markdown 标题；标题里的标签（如推荐等级）转成括号文字
    def heading(m):
        level, inner = m.group(1), m.group(2)
        tags = [strip_tags(t) for t in re.findall(r"<span[^>]*>(.*?)</span>", inner)]
        text = strip_tags(re.sub(r"<span[^>]*>.*?</span>", "", inner))
        if tags:
            text += "（" + "，".join(tags) + "）"
        return "\n\n" + "#" * int(level[1]) + " " + text + "\n\n"

    chunk = re.sub(r"<(h[23])\b[^>]*>(.*?)</\1>", heading, chunk, flags=re.S)

    # 6. 其余 HTML：去掉每行缩进和段内空行，让每段成为一个完整的 HTML 块
    lines = []
    for ln in chunk.split("\n"):
        s = ln.strip()
        if s.startswith("#"):
            lines += ["", s, ""]
        elif s:
            lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. 放回 <pre>
    text = re.sub(r"(\d+)", lambda m: pres[int(m.group(1))], text)
    return text


if __name__ == "__main__":
    main()
