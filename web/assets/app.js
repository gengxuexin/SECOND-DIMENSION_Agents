/* ==========================================================================
   前端交互逻辑。所有数据来自后端 /api/*，前端不做任何业务判定。
   渲染原则：LLM/模型给出的结论一律带来源标记（工具名 / 证据评论 id / 置信度），
             前端只负责如实展示，不做二次推断。
   ========================================================================== */
'use strict';

const API = '';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const PAGE_META = {
  chat:       ['Agent 对话', '用自然语言完成「图像归档 → 标签归一 → 评论洞察 → 运营建议」闭环'],
  dashboard:  ['运营看板', '评论情感分布与库存风险总览'],
  archive:    ['图像归档', 'V1：Function Calling 直调分类服务，返回 Top-K 标签'],
  comments:   ['评论洞察', 'V2：逐条情感/主题打分，按品类聚合并给出证据'],
  governance: ['治理与评测', 'V3：评测回归 + 人工修正回流 + 置信度分流'],
  trace:      ['链路轨迹', '每次工具调用与人工介入的完整留痕'],
  products:   ['商品与标签', '业务上下文：商品表与标准标签库'],
};

let lastState = null;   // 最近一次 WorkflowState，供治理面板/轨迹复用
let chartInst = null;

// ---------------------------------------------------------------- 基础
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error(`响应非 JSON (${res.status}): ${text.slice(0, 120)}`); }
  if (!res.ok && !data.error) throw new Error(`HTTP ${res.status}`);
  return data;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** 极简 Markdown 渲染：只支持标题/表格/列表/加粗，够报告用且不引第三方库。 */
function md(src) {
  const lines = String(src || '').split('\n');
  const out = [];
  let inTable = false, inList = false;
  const closeBlocks = () => {
    if (inTable) { out.push('</tbody></table>'); inTable = false; }
    if (inList) { out.push('</ul>'); inList = false; }
  };
  for (const raw of lines) {
    const l = raw.trimEnd();
    if (/^\s*\|.*\|\s*$/.test(l)) {
      const cells = l.split('|').slice(1, -1).map(c => c.trim());
      if (/^[\s|:-]+$/.test(l)) continue;              // 分隔行
      if (!inTable) {
        closeBlocks();
        out.push('<table><thead><tr>' + cells.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>');
        inTable = true;
      } else {
        out.push('<tr>' + cells.map(c => `<td>${inlineMd(c)}</td>`).join('') + '</tr>');
      }
      continue;
    }
    if (inTable) { out.push('</tbody></table>'); inTable = false; }

    const h = l.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeBlocks(); out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); continue; }

    const li = l.match(/^[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { closeBlocks(); out.push('<ul>'); inList = true; }
      out.push(`<li>${inlineMd(li[1])}</li>`);
      continue;
    }
    if (inList) { out.push('</ul>'); inList = false; }
    out.push(inlineMd(l));
  }
  closeBlocks();
  return out.join('\n');
}

function inlineMd(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

function toast(msg, kind = 'info') {
  const id = 'wb-toast-' + Date.now();
  const color = { info: 'primary', success: 'success', danger: 'danger', warning: 'warning' }[kind];
  const el = document.createElement('div');
  el.id = id;
  el.className = `toast-container position-fixed top-0 end-0 p-3`;
  el.style.zIndex = 2000;
  el.innerHTML = `<div class="toast show text-bg-${color}" role="alert">
    <div class="d-flex"><div class="toast-body">${esc(msg)}</div>
    <button class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div></div>`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function confClass(v) { return v >= 0.8 ? 'conf-high' : v >= 0.6 ? 'conf-mid' : 'conf-low'; }
function pct(v) { return (v * 100).toFixed(0) + '%'; }

// ---------------------------------------------------------------- 路由
function goto(page) {
  $$('.page').forEach(p => p.classList.add('d-none'));
  const target = $('#page-' + page);
  if (target) target.classList.remove('d-none');
  $$('.sidebar-menu .nav-link').forEach(a =>
    a.classList.toggle('active', a.dataset.page === page));
  const [h, s] = PAGE_META[page] || ['', ''];
  $('#pageTitle').textContent = h;
  $('#pageHeading').textContent = h;
  $('#pageSub').textContent = s;
  location.hash = page;
  if (page === 'governance') loadReview();
  if (page === 'trace') loadTraceList();
  if (page === 'dashboard') loadDashboard();
}

// ---------------------------------------------------------------- 侧栏状态
async function loadHealth() {
  try {
    const h = await api('/api/health');
    $('#llmBadge').textContent = h.llm_backend || 'unknown';
    const up = Object.values(h.services || {}).filter(s => s.up).length;
    const total = Object.keys(h.services || {}).length;
    const badge = $('#svcBadge');
    badge.textContent = `${up}/${total} 在线`;
    badge.className = 'badge ' + (up === total ? 'text-bg-success' : 'text-bg-danger');
    $('#svcList').innerHTML = Object.entries(h.services || {}).map(([k, v]) => `
      <div class="d-flex justify-content-between align-items-center mb-1">
        <span class="text-secondary small">${esc(k)}</span>
        <span class="badge text-bg-${v.up ? 'success' : 'danger'}">${v.up ? 'up' : 'down'}</span>
      </div>
      ${v.up ? `<small class="text-secondary d-block" style="font-size:.68rem">${esc(v.backend || '')}</small>` : ''}
    `).join('') || '<small class="text-secondary">无服务信息</small>';

    // 后端可信度：如实展示哪些后端通过了业务数据验证
    const trust = h.backend_trust || {};
    const keys = Object.keys(trust);
    if (keys.length) {
      $('#svcList').insertAdjacentHTML('beforeend', `
        <div class="mt-2 pt-2 border-top border-secondary">
          <small class="text-secondary d-block mb-1">后端可信度（业务数据实测）</small>
          ${keys.map(k => {
            const t = trust[k];
            const cls = t.trusted ? 'success' : 'danger';
            return `<div class="d-flex justify-content-between align-items-center mb-1">
              <span class="text-secondary small">${esc(k)}</span>
              <span class="badge text-bg-${cls}">${(t.measured_acc * 100).toFixed(0)}%</span>
            </div>`;
          }).join('')}
        </div>`);
    }
  } catch (e) {
    $('#svcBadge').className = 'badge text-bg-danger';
    $('#svcBadge').textContent = '离线';
  }
}

async function loadStats() {
  try {
    const d = await api('/api/dashboard');
    $('#statProducts').textContent = d.stats.products;
    $('#statComments').textContent = d.stats.comments;
    $('#statLow').textContent = d.stats.low_stock;
    $('#statTasks').textContent = d.tasks.length;
    const sel = $('#chatScope'), sel2 = $('#insightScope');
    const cats = Object.keys(d.sentiment_by_category || {});
    [sel, sel2].forEach(s => {
      const keep = s.value;
      s.innerHTML = '<option value="all">全店评论</option>' +
        cats.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
      if (cats.includes(keep)) s.value = keep;
    });
  } catch (e) { /* 静默：统计不是关键路径 */ }
}

// ---------------------------------------------------------------- 聊天
function addMsg(who, html) {
  const log = $('#chatLog');
  if (log.querySelector('.text-center')) log.innerHTML = '';
  const isUser = who === 'user';
  const div = document.createElement('div');
  div.className = `msg ${isUser ? 'user' : 'bot'}`;
  div.innerHTML = `
    <div class="avatar"><i class="bi bi-${isUser ? 'person' : 'robot'}"></i></div>
    <div class="bubble">
      <div class="who">${isUser ? '店主' : 'Agent 编排层'}</div>
      <div class="body">${html}</div>
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

async function sendChat() {
  const input = $('#chatInput');
  const q = input.value.trim();
  if (!q) return;
  addMsg('user', esc(q));
  input.value = '';
  $('#sendBtn').disabled = true;

  const bot = addMsg('bot', '<span class="typing"><span></span><span></span><span></span></span>');
  try {
    const st = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ query: q, scope: $('#chatScope').value }),
    });
    lastState = st;
    $('.body', bot).innerHTML = renderReport(st);
    renderTraceMini(st);
    renderGovMini(st);
  } catch (e) {
    $('.body', bot).innerHTML =
      `<div class="text-danger small"><i class="bi bi-exclamation-triangle me-1"></i>调用失败：${esc(e.message)}</div>`;
    toast('调用失败：' + e.message, 'danger');
  } finally {
    $('#sendBtn').disabled = false;
    input.focus();
  }
}

/** 把 WorkflowState 渲染成结构化 HTML（而非直接塞 markdown，便于突出证据）。 */
function renderReport(st) {
  const meta = st._meta || {};
  let h = '';

  h += `<div class="d-flex flex-wrap gap-1 align-items-center mb-2">
    <span class="badge text-bg-primary">意图 ${esc(st.intent)}</span>
    <span class="badge ${st.status === 'done' ? 'text-bg-success' : 'text-bg-warning'}">
      ${esc(st.status)}</span>
    <span class="badge text-bg-light border">${esc(meta.llm_backend || '')}</span>
    ${meta.cost_ms != null ? `<span class="badge text-bg-light border">${meta.cost_ms} ms</span>` : ''}
  </div>`;

  if ((st.errors || []).length) {
    h += `<div class="alert alert-warning py-2 px-3 small mb-2">
      <i class="bi bi-exclamation-triangle me-1"></i>${st.errors.slice(0, 3).map(esc).join('；')}</div>`;
  }

  // 图像归档
  if ((st.image_labels || []).length) {
    h += `<h3><i class="bi bi-image me-1"></i>图像归档</h3><div class="mb-2">`;
    st.image_labels.forEach(t => {
      const c = t.need_confirm ? 'text-bg-warning' : 'text-bg-success';
      h += `<div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
        <span class="badge text-bg-light border">${esc(t.raw)}</span>
        <i class="bi bi-arrow-right"></i>
        <strong>${esc(t.normalized)}</strong>
        <span class="badge ${c}">${(t.conf * 100).toFixed(0)}%</span>
        <span class="text-secondary small">${esc(t.source)}</span>
        ${t.need_confirm ? '<span class="badge text-bg-warning">待人工</span>' : ''}
      </div>`;
    });
    h += `</div>`;
  }

  // 评论洞察
  if ((st.comment_results || {}) && Object.keys(st.comment_results).length) {
    const n = Object.keys(st.comment_results).length;
    const dist = countSentiment(st.comment_results);
    h += `<h3><i class="bi bi-chat-square-text me-1"></i>评论洞察</h3>
      <div class="mb-2 small">分析 ${n} 条：
      <span class="text-success">正面 ${dist.positive}</span> /
      <span class="text-danger">负面 ${dist.negative}</span> /
      <span class="text-secondary">中性 ${dist.neutral}</span>
      （负面占比 ${pct(dist.negRate)}）</div>`;
    if ((st.themes || []).length) {
      h += '<table class="table table-sm"><thead><tr><th>主题</th><th>品类</th><th class="text-center">条数</th><th>倾向</th><th>证据</th></tr></thead><tbody>';
      st.themes.slice(0, 8).forEach(t => {
        const sc = { negative: 'danger', positive: 'success', neutral: 'secondary' }[t.sentiment];
        h += `<tr><td>${esc(t.theme.split('-').pop())}</td><td>${esc(t.category)}</td>
          <td class="text-center">${t.count}</td>
          <td><span class="badge text-bg-${sc}">${t.sentiment}</span></td>
          <td>${(t.evidence_ids || []).slice(0, 5).map(e => `<span class="evid">${esc(e)}</span>`).join('')}</td></tr>`;
      });
      h += '</tbody></table>';
    }
  }

  // 运营建议
  if ((st.advice || []).length) {
    h += `<h3><i class="bi bi-lightbulb me-1"></i>运营建议</h3>`;
    st.advice.forEach(a => {
      h += `<div class="card mb-2 advice-card p-${esc(a.priority)}">
        <div class="card-body py-2 px-3">
          <div class="d-flex justify-content-between align-items-start gap-2">
            <strong class="small">${esc(a.title)}</strong>
            <span class="badge text-bg-light border">${esc(a.priority)}</span>
          </div>
          <div class="small mt-1">${esc(a.detail)}</div>
          <div class="mt-1 d-flex flex-wrap align-items-center gap-1">
            <span class="text-secondary small">置信度</span>
            <div class="conf-bar" style="width:70px"><span class="${confClass(a.confidence)}" style="width:${a.confidence * 100}%"></span></div>
            <small class="text-secondary">${pct(a.confidence)}</small>
            <span class="badge text-bg-secondary">${esc(a.action_type)}</span>
          </div>
          <div class="mt-1">
            ${(a.tool_refs || []).map(t => `<span class="tool-ref"><i class="bi bi-tools me-1"></i>${esc(t)}</span>`).join('')}
            ${(a.evidence_ids || []).map(e => `<span class="evid">${esc(e)}</span>`).join('')}
            ${a.need_human_confirm ? '<span class="badge text-bg-warning">需人工确认</span>' : ''}
          </div>
        </div></div>`;
    });
  }

  if (!h) h = '<div class="text-secondary small">本次任务未产生结构化产出。</div>';
  return h;
}

function countSentiment(results) {
  let positive = 0, negative = 0, neutral = 0;
  Object.values(results).forEach(s => {
    if (s.sentiment === 'positive') positive++;
    else if (s.sentiment === 'negative') negative++;
    else neutral++;
  });
  const n = Math.max(positive + negative + neutral, 1);
  return { positive, negative, neutral, negRate: negative / n };
}

// ---------------------------------------------------------------- 轨迹 / 治理
function renderTraceMini(st) {
  const items = (st.trace || []).filter(e =>
    ['route', 'plan', 'tool_call', 'human_gate', 'finish'].includes(e.step));
  $('#traceMini').innerHTML = items.map(e => `
    <li class="list-group-item d-flex justify-content-between align-items-center py-2">
      <div>
        <span class="badge text-bg-light border me-1">${esc(e.step)}</span>
        <span class="small">${esc(e.detail || '')}</span>
      </div>
      <small class="text-secondary text-nowrap ms-2">${e.actor}${e.cost_ms ? ' ' + e.cost_ms + 'ms' : ''}</small>
    </li>`).join('') || '<li class="list-group-item text-secondary small">无轨迹</li>';
}

function renderGovMini(st) {
  const auto = Object.values(st.comment_results || {}).filter(s => s.sentiment_conf >= 0.6).length;
  const total = Object.keys(st.comment_results || {}).length;
  const lowConf = (st.image_labels || []).filter(t => t.need_confirm).length;
  const noEv = (st.advice || []).filter(a => !(a.evidence_ids || []).length).length;
  const rate = total ? auto / total : (st.advice || []).length ? 1 : 0;

  $('#govMini').innerHTML = `
    <div class="d-flex justify-content-between small mb-1">
      <span class="text-secondary">自动采信率</span><strong>${pct(rate)}</strong></div>
    <div class="conf-bar mb-3"><span class="${confClass(rate)}" style="width:${rate * 100}%"></span></div>
    <div class="small text-secondary">
      <div class="d-flex justify-content-between"><span>待人工标签</span><span>${lowConf}</span></div>
      <div class="d-flex justify-content-between"><span>无证据建议</span><span class="${noEv ? 'text-danger' : ''}">${noEv}</span></div>
      <div class="d-flex justify-content-between"><span>证据闭环</span>
        <span class="text-success">${noEv ? '未通过' : '通过'}</span></div>
    </div>
    <div class="text-secondary mt-2" style="font-size:.72rem">
      低置信结果不自动采信，转人工确认队列</div>`;
}

// ---------------------------------------------------------------- 图像归档
let uploadedPaths = [];

async function doClassify() {
  const btn = $('#classifyBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>识别中';
  $('#classifyResult').innerHTML = '<div class="text-secondary small">调用 classify_image 工具…</div>';
  try {
    const r = await api('/api/classify', {
      method: 'POST',
      body: JSON.stringify({ image_paths: uploadedPaths, top_k: +$('#topK').value }),
    });
    let h = `<div class="alert alert-secondary py-2 px-3 small">
      编排后端 <strong>${esc(r.llm)}</strong>　工具调用 <strong>${r.calls.length}</strong> 次</div>`;
    r.calls.forEach(c => {
      const badge = c.ok ? 'text-bg-success' : 'text-bg-danger';
      h += `<div class="border rounded p-2 mb-2">
        <div class="d-flex justify-content-between align-items-center">
          <code class="small">${esc(c.name)}(${esc(JSON.stringify(c.args).slice(0, 60))})</code>
          <span class="badge ${badge}">${esc(c.summary)} · ${c.cost_ms}ms</span>
        </div>`;
      if (c.result && c.result.predictions) {
        h += '<div class="mt-2">';
        c.result.predictions.forEach(p => {
          const cc = p.cross_check || {};
          h += `<div class="d-flex align-items-center gap-2 mb-1 flex-wrap">
            <span class="badge text-bg-light border">#${p.top_k_rank}</span>
            <strong class="small">${esc(p.label)}</strong>
            <div class="conf-bar" style="width:90px"><span class="${confClass(p.score)}" style="width:${p.score * 100}%"></span></div>
            <small class="text-secondary">${pct(p.score)}</small>
            ${cc.aux_backend ? `<span class="badge ${cc.agree ? 'text-bg-success' : 'text-bg-warning'}">
              <i class="bi bi-${cc.agree ? 'check2-circle' : 'exclamation-triangle'} me-1"></i>
              交叉验证 ${esc(cc.aux_backend)}=${esc(cc.aux_top1)}</span>` : ''}
            ${p.need_confirm ? '<span class="badge text-bg-warning">需人工确认</span>' : ''}
          </div>`;
          if (p.conflict_reason) {
            h += `<div class="text-secondary small ms-4" style="font-size:.72rem">${esc(p.conflict_reason)}</div>`;
          }
        });
        h += '</div>';
      }
      if (c.result && c.result.error) {
        h += `<div class="text-danger small mt-1">${esc(c.result.error.code)}: ${esc(c.result.error.message)}</div>`;
      }
      h += '</div>';
    });
    $('#classifyResult').innerHTML = h;
  } catch (e) {
    $('#classifyResult').innerHTML =
      `<div class="alert alert-danger py-2 small">识别失败：${esc(e.message)}<br>
       <small>请确认模型服务已启动：<code>python run.py serve</code></small></div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-magic me-1"></i>开始识别';
  }
}

function previewUploads(files) {
  uploadedPaths = [];
  const box = $('#uploadPreview');
  box.innerHTML = '';
  [...files].forEach(f => {
    const url = URL.createObjectURL(f);
    box.insertAdjacentHTML('beforeend',
      `<img src="${url}" class="upthumb" title="${esc(f.name)}">`);
  });
}

// ---------------------------------------------------------------- 评论洞察
async function runInsight() {
  const scope = $('#insightScope').value;
  $('#insightResult').innerHTML =
    '<div class="text-secondary small"><span class="spinner-border spinner-border-sm me-1"></span>运行多 Agent 闭环…</div>';
  try {
    const st = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ query: `看看${scope === 'all' ? '全店' : scope}评论里尺码、颜色、质量有什么问题`,
                             scope }),
    });
    lastState = st;
    $('#insightResult').innerHTML = renderReport(st);
    loadStats();
  } catch (e) {
    $('#insightResult').innerHTML =
      `<div class="alert alert-danger py-2 small">运行失败：${esc(e.message)}</div>`;
  }
}

// ---------------------------------------------------------------- 评测
async function runEval() {
  $('#evalResult').innerHTML =
    '<div class="text-secondary small"><span class="spinner-border spinner-border-sm me-1"></span>评测中…</div>';
  try {
    const r = await api('/api/eval', { method: 'POST', body: '{}' });
    const kinds = Object.entries(r.by_kind || {});
    let h = `<div class="d-flex align-items-center gap-2 mb-3">
      <span class="badge ${r.overall >= 0.8 ? 'text-bg-success' : 'text-bg-danger'}">
        总准确率 ${pct(r.overall)}</span>
      <span class="badge text-bg-light border">${r.passed}/${r.total} 通过</span>
      <span class="badge text-bg-light border">${esc(r.llm_backend)}</span></div>`;
    h += '<table class="table table-sm"><thead><tr><th>用例类型</th><th class="text-center">通过</th><th class="text-center">总数</th><th class="text-end">准确率</th><th class="text-end">均耗时</th></tr></thead><tbody>';
    kinds.forEach(([k, b]) => {
      h += `<tr><td><code>${esc(k)}</code></td><td class="text-center">${b.pass}</td>
        <td class="text-center">${b.n}</td>
        <td class="text-end"><span class="badge ${b.accuracy >= 0.8 ? 'text-bg-success' : 'text-bg-danger'}">${pct(b.accuracy)}</span></td>
        <td class="text-end text-secondary small">${b.avg_cost_ms}ms</td></tr>`;
    });
    h += '</tbody></table>';
    const fails = (r.details || []).filter(d => !d.pass);
    if (fails.length) {
      h += `<div class="alert alert-warning py-2 px-3 small mb-0">
        <strong>未通过用例</strong><ul class="mb-0 mt-1">` +
        fails.map(f => `<li>[${esc(f.id)}] ${esc(f.input)} → ${esc(f.note)}</li>`).join('') +
        '</ul></div>';
    } else {
      h += '<div class="alert alert-success py-2 px-3 small mb-0">全部用例通过 ✓</div>';
    }
    $('#evalResult').innerHTML = h;
    toast(`评测完成：${pct(r.overall)}`, r.overall >= 0.8 ? 'success' : 'warning');
  } catch (e) {
    $('#evalResult').innerHTML =
      `<div class="alert alert-danger py-2 small">评测失败：${esc(e.message)}</div>`;
  }
}

async function loadReview() {
  try {
    const r = await api('/api/review');
    const rows = r.samples || [];
    $('#sampleBody').innerHTML = rows.length ? rows.slice(-15).reverse().map(s => `
      <tr><td><span class="badge text-bg-light border">${esc(s.kind)}</span></td>
        <td><span class="badge text-bg-${s.decision === 'accept' ? 'success' : 'warning'}">${esc(s.decision)}</span></td>
        <td class="small">${esc(s.corrected_value || '')}</td></tr>`).join('')
      : '<tr><td colspan="3" class="text-secondary small">暂无回流样本</td></tr>';
  } catch (e) { /* 忽略 */ }
}

// ---------------------------------------------------------------- 轨迹页
async function loadTraceList() {
  try {
    const r = await api('/api/history');
    const sel = $('#traceSelect');
    if (!(r.tasks || []).length) {
      sel.innerHTML = '<option>暂无任务</option>';
      return;
    }
    sel.innerHTML = r.tasks.map(t =>
      `<option value="${esc(t.task_id)}">${esc(t.task_id)} · ${esc(t.intent)} · ${esc(t.status)}</option>`).join('');
    showTrace(sel.value);
  } catch (e) { /* 忽略 */ }
}

async function showTrace(taskId) {
  if (!taskId) return;
  try {
    const d = await api('/api/task/' + taskId);
    const items = (d.trace || []);
    let h = `<div class="mb-2 d-flex flex-wrap gap-1">
      <span class="badge text-bg-primary">${esc(d.intent)}</span>
      <span class="badge text-bg-light border">${esc(d.status)}</span>
      <span class="badge text-bg-light border">${esc(d.task_id)}</span></div>`;
    h += '<div class="tl">';
    items.forEach(e => {
      h += `<div class="tl-item act-${esc(e.actor)}">
        <div class="tl-step">${esc(e.step)} · ${esc(e.actor)}${e.cost_ms ? ' · ' + e.cost_ms + 'ms' : ''}</div>
        <div class="tl-detail">${esc(e.detail || '')}</div>
      </div>`;
    });
    h += '</div>';
    $('#traceDetail').innerHTML = h;
  } catch (e) {
    $('#traceDetail').innerHTML =
      `<div class="alert alert-warning py-2 small">${esc(e.message)}</div>`;
  }
}

// ---------------------------------------------------------------- 看板
async function loadDashboard() {
  try {
    const d = await api('/api/dashboard');
    const byCat = d.sentiment_by_category || {};
    const cats = Object.keys(byCat);
    $('#lowStockBody').innerHTML = (d.low_stock || []).map(p => `
      <tr><td class="small">${esc(p.name)}</td><td class="small">${esc(p.category)}</td>
        <td class="text-end"><span class="badge text-bg-warning">${p.stock}</span></td>
        <td class="text-end small">¥${p.price}</td></tr>`).join('')
      || '<tr><td colspan="4" class="text-secondary small">无低库存商品</td></tr>';

    if (chartInst) chartInst.destroy();
    const cv = $('#chartSentiment');
    if (cv) {
      chartInst = new Chart(cv, {
        type: 'bar',
        data: {
          labels: cats,
          datasets: [
            { label: '正面', data: cats.map(c => byCat[c].positive || 0), backgroundColor: '#198754' },
            { label: '负面', data: cats.map(c => byCat[c].negative || 0), backgroundColor: '#dc3545' },
            { label: '中性', data: cats.map(c => byCat[c].neutral || 0), backgroundColor: '#adb5bd' },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
          plugins: { legend: { position: 'bottom' } },
        },
      });
    }
  } catch (e) { /* 忽略 */ }
}

// ---------------------------------------------------------------- 商品
async function loadProducts() {
  try {
    const r = await api('/api/products');
    $('#productBody').innerHTML = (r.products || []).map(p => `
      <tr><td><code>${esc(p.sku_id)}</code></td><td>${esc(p.name)}</td>
        <td><span class="badge text-bg-light border">${esc(p.category)}</span></td>
        <td class="small">${esc(p.style)}</td><td class="small">${esc(p.color)}</td>
        <td class="text-end">¥${p.price}</td>
        <td class="text-end">${p.stock <= 12 ? `<span class="badge text-bg-warning">${p.stock}</span>` : p.stock}</td>
        <td class="small text-secondary">${esc(p.listed_at)}</td></tr>`).join('');

    const t = await api('/api/tags');
    const lib = t.library || {};
    const canon = Object.keys(lib.canonical || {});
    const alias = lib.aliases || {};
    $('#tagCloud').innerHTML =
      `<div class="mb-2"><strong class="small">标准标签（${canon.length}）</strong><br>` +
      canon.map(c => `<span class="tag-pill">${esc(c)}</span>`).join('') + `</div>
      <div><strong class="small">别名映射（${Object.keys(alias).length}）</strong><br>` +
      Object.entries(alias).slice(0, 40).map(([a, b]) =>
        `<span class="tag-pill">${esc(a)} <i class="bi bi-arrow-right"></i> ${esc(b)}</span>`).join('') +
      `</div>`;
  } catch (e) { /* 忽略 */ }
}

// ---------------------------------------------------------------- 绑定
document.addEventListener('DOMContentLoaded', () => {
  // 导航
  $$('.sidebar-menu .nav-link[data-page]').forEach(a =>
    a.addEventListener('click', e => { e.preventDefault(); goto(a.dataset.page); }));
  $('#refreshBtn').addEventListener('click', () => location.reload());

  // 聊天
  $('#sendBtn').addEventListener('click', sendChat);
  $('#chatInput').addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
  $('#clearChat').addEventListener('click', () => {
    $('#chatLog').innerHTML = `<div class="text-center text-secondary small py-5">
      <i class="bi bi-stars fs-2 d-block mb-2"></i>试试：「根据评论和库存给我一些选品建议」</div>`;
    $('#traceMini').innerHTML = '<li class="list-group-item text-secondary small">等待任务…</li>';
    $('#govMini').innerHTML = '<div class="text-secondary small">等待任务…</div>';
  });
  $$('.quick-q').forEach(b => b.addEventListener('click', () => {
    $('#chatInput').value = b.dataset.q;
    sendChat();
  }));

  // 图像归档
  $('#imgInput').addEventListener('change', async e => {
    const files = [...e.target.files];
    if (!files.length) return;
    previewUploads(files);
    uploadedPaths = [];
    for (const f of files) {
      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          headers: { 'X-Filename': encodeURIComponent(f.name) },
          body: f,
        });
        const d = await res.json();
        if (d.ok) uploadedPaths.push(d.path);
        else toast(`上传失败：${d.error?.message || ''}`, 'danger');
      } catch (err) { toast('上传异常：' + err.message, 'danger'); }
    }
    toast(`已上传 ${uploadedPaths.length} 张，可点击识别`, 'success');
  });
  $('#classifyBtn').addEventListener('click', doClassify);

  // 评论洞察
  $('#insightBtn').addEventListener('click', runInsight);

  // 评测
  $('#evalBtn').addEventListener('click', runEval);

  // 轨迹
  $('#traceSelect').addEventListener('change', e => showTrace(e.target.value));

  // 时钟
  setInterval(() => {
    $('#clock').innerHTML = '<i class="bi bi-clock"></i> ' +
      new Date().toLocaleTimeString('zh-CN', { hour12: false });
  }, 1000);

  // 初始
  const page = location.hash.replace('#', '') || 'chat';
  goto(PAGE_META[page] ? page : 'chat');
  loadHealth();
  loadStats();
  loadProducts();
  setInterval(loadHealth, 20000);

  // 演示模式：?demo=1 自动跑一次对话（用于截图 / 现场演示 / 作品集展示）
  const params = new URLSearchParams(location.search);
  if (params.get('demo') === '1') {
    const q = params.get('q') || '根据评论和库存给我一些选品建议';
    $('#chatInput').value = q;
    setTimeout(() => sendChat(), 600);
  }
});