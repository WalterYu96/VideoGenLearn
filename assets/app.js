(() => {
  'use strict';

  const CH = window.CHAPTERS;
  const byId = Object.fromEntries(CH.map((c) => [c.id, c]));
  const mdCache = new Map();
  const $ = (s, el = document) => el.querySelector(s);

  // ---------- markdown + math ----------
  marked.setOptions({ gfm: true, breaks: false });

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function renderMath(tex, display) {
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false }); }
    catch (e) { return `<code>${escapeHtml(tex)}</code>`; }
  }

  // 把公式和自定义块先抽出来，避免被 marked 改写
  function protect(md) {
    const slots = [];
    const put = (html) => { slots.push(html); return `\uE000${slots.length - 1}\uE001`; };

    // ```flow 流程块：每行一个步骤，"  ↳ " 开头为注释
    md = md.replace(/```flow\n([\s\S]*?)```/g, (_, body) => {
      const steps = body.trim().split('\n').filter(Boolean).map((line) => {
        const [main, ...notes] = line.split(' | ');
        const note = notes.length ? `<span class="flow-note">${escapeHtml(notes.join(' | '))}</span>` : '';
        return `<div class="flow-step"><span class="flow-main">${escapeHtml(main.trim())}</span>${note}</div>`;
      });
      return put(`<div class="flow">${steps.join('<div class="flow-arrow">↓</div>')}</div>`);
    });

    // 普通代码块保持原样（防止其中的 $ 被当成公式）
    md = md.replace(/```[\s\S]*?```/g, (m) => put(marked.parse(m)));
    md = md.replace(/`[^`\n]+`/g, (m) => put(marked.parseInline(m)));

    // callout 标题行单独成段：> [!结论] 标题  →  > [!结论|标题]
    md = md.replace(/^> \[!([^\]|]+)\][ \t]*(.*)$/gm, (_, k, t) => `> [!${k}|${t.trim()}]\n>`);

    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => put(`<div class="math-block">${renderMath(tex.trim(), true)}</div>`));
    md = md.replace(/\$([^$\n]+?)\$/g, (_, tex) => put(renderMath(tex, false)));
    return { md, slots };
  }

  function restore(html, slots) {
    // 占位符可能被 marked 包进 <p>
    return html
      .replace(/<p>\uE000(\d+)\uE001<\/p>/g, (_, i) => slots[+i])
      .replace(/\uE000(\d+)\uE001/g, (_, i) => slots[+i]);
  }

  function renderMarkdown(src) {
    const { md, slots } = protect(src);
    const html = restore(marked.parse(md), slots);
    const tpl = document.createElement('template');
    tpl.innerHTML = html;
    const root = tpl.content;

    // callout: > [!结论] 标题
    root.querySelectorAll('blockquote').forEach((bq) => {
      const p = bq.firstElementChild;
      if (!p) return;
      const m = p.innerHTML.match(/^\[!([^\]|]+)(?:\|([^\]]*))?\]\s*/);
      if (!m) return;
      const kind = m[1];
      const title = (m[2] || '').trim();
      p.innerHTML = p.innerHTML.slice(m[0].length);
      if (!p.innerHTML.trim()) p.remove();
      const cls = { '结论': 'key', '提示': 'tip', '注意': 'warn', '经验': 'tip', '来源': 'note' }[kind] || 'note';
      bq.classList.add('callout', `callout-${cls}`);
      const label = document.createElement('div');
      label.className = 'callout-label';
      label.textContent = title || kind;
      bq.prepend(label);
    });

    // 表格横向滚动容器
    root.querySelectorAll('table').forEach((t) => {
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      t.replaceWith(wrap);
      wrap.appendChild(t);
    });

    root.querySelectorAll('a[href^="http"]').forEach((a) => {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
    return root;
  }

  async function loadMd(file) {
    if (mdCache.has(file)) return mdCache.get(file);
    const res = await fetch(file, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${res.status} ${file}`);
    const text = await res.text();
    mdCache.set(file, text);
    return text;
  }

  // ---------- sidebar ----------
  function buildNav(activeId) {
    const nav = $('#nav');
    nav.innerHTML = '';
    const home = document.createElement('li');
    home.innerHTML = `<a href="#/" class="${activeId ? '' : 'active'}"><span class="num">⌂</span><span>首页</span></a>`;
    nav.appendChild(home);
    CH.forEach((c) => {
      const li = document.createElement('li');
      if (c.parent) li.className = 'sub';
      li.innerHTML = `<a href="#/${c.id}" class="${c.id === activeId ? 'active' : ''}">
        <span class="num">${c.num}</span><span class="t">${escapeHtml(c.title)}</span></a>`;
      nav.appendChild(li);
    });
  }

  // ---------- toc + scrollspy ----------
  let spy;
  function buildToc(article, chapterId) {
    const toc = $('#toc');
    const hs = [...article.querySelectorAll('h2, h3')];
    if (spy) spy.disconnect();
    if (!hs.length) { toc.innerHTML = ''; return; }
    toc.innerHTML = '<div class="toc-title">本页内容</div>' + hs.map((h) =>
      `<a href="#/${chapterId}/${h.id}" data-id="${h.id}" class="${h.tagName === 'H3' ? 'lvl3' : ''}">${escapeHtml(h.textContent.replace(/#$/, ''))}</a>`
    ).join('');
    const links = new Map([...toc.querySelectorAll('a')].map((a) => [a.dataset.id, a]));
    spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove('active'));
          const a = links.get(e.target.id);
          if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-70px 0px -70% 0px' });
    hs.forEach((h) => spy.observe(h));
  }

  function assignIds(article) {
    let i = 0;
    article.querySelectorAll('h2, h3').forEach((h) => {
      h.id = `s${++i}`;
      const a = document.createElement('a');
      a.className = 'anchor';
      a.textContent = '#';
      a.setAttribute('aria-hidden', 'true');
      h.appendChild(a);
    });
  }

  // ---------- pages ----------
  function renderHome() {
    document.title = '视频生成学习站';
    const content = $('#content');
    const stages1 = [
      ['data', '数据', '切镜 · 过滤 · 去重 · caption'],
      ['architecture', '预训练架构', '3D VAE · DiT/MMDiT · Flow Matching'],
      ['midtraining', '中训练', '质量退火 · 能力扩展 · 任务注入'],
      ['posttraining', 'SFT', '小而精 · 类别均衡 · 模型融合'],
    ];
    const stages2 = [
      ['posttraining', '奖励与 RL', 'DPO · GRPO · ReFL · 多奖励'],
      ['posttraining', '蒸馏', '步数 + CFG 蒸馏 · 自回归蒸馏'],
      ['posttraining', 'OPD', '在线策略蒸馏 · 多任务合并'],
      ['inference', '推理效率', '稀疏注意力 · 量化 · 缓存'],
    ];
    const stage = (row) => row.map(([id, t, d], i) =>
      `<a class="stage" href="#/${id}"><span class="stage-i">${i + 1}</span><b>${t}</b><small>${d}</small></a>`
    ).join('<span class="stage-arrow">→</span>');

    const cards = CH.filter((c) => !c.parent).map((c) => {
      const subs = CH.filter((s) => s.parent === c.id);
      const subHtml = subs.length ? `<div class="card-subs">${subs.map((s) =>
        `<a href="#/${s.id}">${s.num} ${escapeHtml(s.title.replace('专题：', ''))}</a>`).join('')}</div>` : '';
      return `<div class="card">
        <a class="card-main" href="#/${c.id}"><span class="card-num">${c.num}</span>
        <span class="card-title">${escapeHtml(c.title)}</span></a>${subHtml}</div>`;
    }).join('');

    content.innerHTML = `
      <section class="hero">
        <p class="eyebrow">视频生成技术学习手册 · 2026-11</p>
        <h1>从文生图到视频生成：一条完整的学习路径</h1>
        <p class="lede">2026 年头部视频模型的主干已基本统一为 <b>3D VAE + DiT/MMDiT + Flow Matching</b>，
        竞争重点转到<b>数据</b>、<b>后训练（RL / 蒸馏 / OPD）</b>和<b>推理成本</b>。本站按全流程 8 个阶段组织，
        附带数据管线专题（镜头切分、去重、各家管线对比）。</p>
        <div class="hero-actions">
          <a class="btn primary" href="#/overview">从导读开始</a>
          <a class="btn" href="#/roadmap">查看 4 周学习路线</a>
        </div>
      </section>
      <section>
        <h2 class="home-h">数据管线专题</h2>
        <div class="features">
          <a class="feature wide" href="#/data-industry">
            <span class="feature-tag">重点</span>
            <b>顶级团队视频数据管线</b>
            <small>Seedance、Wan、Kling、MiniMax H3 逐家拆解，另附 HunyuanVideo 1.5、Waver、Open-Sora 2.0、Koala-36M 开源管线对比</small>
          </a>
          <a class="feature" href="#/data-shot"><b>镜头切分</b><small>PySceneDetect、TransNetV2、AutoShot 与 embedding 校验</small></a>
          <a class="feature" href="#/data-dedup"><b>视频与图像去重</b><small>感知哈希、SSCD、ViSiL，以及 EmbeddingGemma 2 的适用性</small></a>
        </div>
      </section>
      <section>
        <h2 class="home-h">视频模型全流程 · 8 个阶段</h2>
        <p class="muted">第一行是“能力”阶段，第二行是“对齐与成本”阶段。T2I 经验在数据和 RL 两处最能直接复用。</p>
        <div class="stages"><div class="stage-row"><span class="row-label">能力</span>${stage(stages1)}</div>
        <div class="stage-row"><span class="row-label">对齐与成本</span>${stage(stages2)}</div></div>
      </section>
      <section>
        <h2 class="home-h">章节</h2>
        <div class="cards">${cards}</div>
      </section>`;
    $('#toc').innerHTML = '';
    $('#chapterFooter').innerHTML = '';
    buildNav(null);
  }

  async function renderChapter(id, anchor) {
    const c = byId[id];
    const content = $('#content');
    if (!c) { content.innerHTML = '<h1>没有找到这个章节</h1><p><a href="#/">回到首页</a></p>'; return; }
    document.title = `${c.title} · 视频生成学习站`;
    buildNav(id);
    let src;
    try {
      src = await loadMd(c.file);
    } catch (e) {
      content.innerHTML = `<div class="callout callout-warn"><div class="callout-label">无法加载</div>
        <p>读取 <code>${c.file}</code> 失败。直接双击打开 index.html 时浏览器会禁止读取本地文件，请在
        VideoGenLearn 目录运行 <code>python3 -m http.server 8765</code> 后访问 http://localhost:8765 。</p></div>`;
      return;
    }
    const frag = renderMarkdown(src);
    content.innerHTML = '';
    if (c.parent) {
      const p = byId[c.parent];
      const crumb = document.createElement('div');
      crumb.className = 'crumb';
      crumb.innerHTML = `<a href="#/${p.id}">${p.num}. ${escapeHtml(p.title)}</a> / 专题`;
      content.appendChild(crumb);
    }
    content.appendChild(frag);
    assignIds(content);
    buildToc(content, id);
    renderFooter(c);

    if (anchor) {
      const el = document.getElementById(anchor);
      if (el) { el.scrollIntoView(); return; }
    }
    window.scrollTo(0, 0);
  }

  function renderFooter(c) {
    const i = CH.indexOf(c);
    const prev = CH[i - 1], next = CH[i + 1];
    $('#chapterFooter').innerHTML = `
      <div class="pager">
        ${prev ? `<a class="pager-link" href="#/${prev.id}"><small>上一章</small><span>${prev.num} ${escapeHtml(prev.title)}</span></a>` : '<span></span>'}
        ${next ? `<a class="pager-link next" href="#/${next.id}"><small>下一章</small><span>${next.num} ${escapeHtml(next.title)}</span></a>` : ''}
      </div>`;
  }

  // ---------- router ----------
  function route() {
    const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
    closeMenu();
    hideResults();
    if (!parts.length) renderHome();
    else renderChapter(parts[0], parts[1]);
  }

  // ---------- search ----------
  let index = null;
  async function buildIndex() {
    if (index) return index;
    const out = [];
    await Promise.all(CH.map(async (c) => {
      let src;
      try { src = await loadMd(c.file); } catch (e) { return; }
      // 与 assignIds 的编号规则一致：按 h2/h3 顺序编号
      const lines = src.replace(/```[\s\S]*?```/g, (m) => m.replace(/^#/gm, ' ')).split('\n');
      let n = 0, cur = { chapter: c, anchor: '', title: c.title, text: '' };
      out.push(cur);
      lines.forEach((ln) => {
        const m = ln.match(/^(#{2,3})\s+(.*)/);
        if (m) {
          n += 1;
          cur = { chapter: c, anchor: `s${n}`, title: m[2].replace(/[*`]/g, ''), text: '' };
          out.push(cur);
        } else {
          cur.text += ' ' + ln;
        }
      });
    }));
    out.forEach((s) => {
      s.plain = s.text.replace(/\$[^$]*\$/g, ' ').replace(/[|*#>`\[\]()_-]+/g, ' ').replace(/\s+/g, ' ').trim();
    });
    index = out;
    return out;
  }

  function hideResults() { $('#searchResults').hidden = true; }

  async function doSearch(q) {
    const box = $('#searchResults');
    q = q.trim().toLowerCase();
    if (!q) { hideResults(); return; }
    const idx = await buildIndex();
    const terms = q.split(/\s+/);
    const hits = [];
    idx.forEach((s) => {
      const hay = (s.title + ' ' + s.plain).toLowerCase();
      if (!terms.every((t) => hay.includes(t))) return;
      const score = terms.reduce((a, t) => a + (s.title.toLowerCase().includes(t) ? 5 : 0) + hay.split(t).length - 1, 0);
      const pos = s.plain.toLowerCase().indexOf(terms[0]);
      const start = Math.max(0, pos - 30);
      let snip = s.plain.slice(start, start + 110);
      snip = escapeHtml(snip);
      terms.forEach((t) => {
        const re = new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        snip = snip.replace(re, (m) => `<mark>${m}</mark>`);
      });
      hits.push({ s, score, snip: (start > 0 ? '…' : '') + snip + '…' });
    });
    hits.sort((a, b) => b.score - a.score);
    box.innerHTML = hits.length ? hits.slice(0, 20).map(({ s, snip }) =>
      `<a href="#/${s.chapter.id}${s.anchor ? '/' + s.anchor : ''}"><b>${s.chapter.num} · ${escapeHtml(s.title)}</b><span>${snip}</span></a>`
    ).join('') : '<div class="empty">没有匹配的内容</div>';
    box.hidden = false;
  }

  // ---------- chrome ----------
  function closeMenu() { document.body.classList.remove('menu-open'); }

  function initChrome() {
    $('#menuBtn').onclick = () => document.body.classList.toggle('menu-open');
    $('#scrim').onclick = closeMenu;
    $('#themeBtn').onclick = () => {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') ||
        (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('vgl-theme', next); } catch (e) {}
    };
    const input = $('#search');
    let timer;
    input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => doSearch(input.value), 120); });
    input.addEventListener('focus', () => { buildIndex(); if (input.value) doSearch(input.value); });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { input.blur(); hideResults(); }
      if (e.key === 'Enter') { const a = $('#searchResults a'); if (a) { location.hash = a.getAttribute('href'); input.blur(); } }
    });
    document.addEventListener('click', (e) => { if (!e.target.closest('.search-wrap')) hideResults(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== input) { e.preventDefault(); input.focus(); }
    });
  }

  initChrome();
  window.addEventListener('hashchange', route);
  route();
})();
