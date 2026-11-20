"use strict";

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = (value) => Number(value || 0).toLocaleString("en-US");
const statusNames = {running:"调查进行中",complete:"已完成",partial:"部分完成",failed:"运行失败",interrupted:"已中断",pending:"准备中",planned:"计划中",in_progress:"模型报告进行中",completed:"已返回",blocked:"阻塞 / 受限"};
const kinds = {task:"研究问题",plan:"计划",gap:"证据缺口",followup:"补充调查",temporal:"时间调查",source_check:"来源追溯",synthesis:"报告与停止",search:"搜索",read:"阅读原文",claim:"保存主张",verify:"核验"};
const icon = name => AtlasIcons.svg(name);
const icons = Object.fromEntries(Object.entries({task:"flame",plan:"plan",gap:"unknown",followup:"branch",temporal:"timeline",source_check:"source",synthesis:"report",search:"search",read:"read",claim:"point",verify:"verify"}).map(([kind,name])=>[kind,icon(name)]));
let data = null, selectedRun = null, csrf = "", activeTab = "events", selectedNode = null;
let replayValue = null, playback = null, parentRun = null, pollBusy = false, lastVersion = "", lastToast = null, lastRunList = "";

async function api(path, options = {}) {
  const response = await fetch(path, {cache:"no-store", ...options, headers:{"Content-Type":"application/json","X-Research-Token":csrf,...options.headers}});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload;
}
function toast(message) {
  clearTimeout(lastToast); $("#toast").textContent = message; $("#toast").hidden = false;
  lastToast = setTimeout(() => {$("#toast").hidden = true;}, 5000);
}
async function health() {
  const result = await api("/api/health"); csrf = result.csrf_token;
  $("#connection").textContent = result.ready ? "DeepSeek + Tavily 已就绪" : "本地浏览 · API 配置未齐";
  $("#connection").title = result.model;
  if (result.live_budget) {
    const limits = result.live_budget;
    const fields = [limits.max_tokens == null ? null : `${num(limits.max_tokens)} token`,
      limits.max_model_calls == null ? null : `${num(limits.max_model_calls)} 次模型请求`,
      limits.deadline_seconds == null ? null : `${num(limits.deadline_seconds / 60)} 分钟`];
    $(".run-budget").textContent = fields.filter(Boolean).join(" · ") || "无累计上限 · 按实际调用记账";
  }
  return result;
}
async function loadRuns(pick = false) {
  const result = await api("/api/runs");
  const runListHtml = result.runs.map(run => `<button class="run-item ${selectedRun === run.id ? "active" : ""}" data-run="${esc(run.id)}"><span class="run-item-title">${esc(run.question)}</span><span class="run-item-meta"><i class="marker ${esc(run.status)}"></i>${esc(run.mock ? "合成 / 脚本模拟" : run.condition.toUpperCase())}<span>${esc(statusNames[run.status] || run.status)}</span></span></button>`).join("") || '<p class="muted">尚无调查记录</p>';
  if(runListHtml!==lastRunList){$("#run-list").innerHTML=runListHtml;lastRunList=runListHtml;}
  if (pick && !selectedRun && result.runs.length) {
    const requested = new URLSearchParams(location.search).get("run");
    await chooseRun(result.runs.find(run => run.id === requested)?.id || result.runs[0].id);
  }
}
async function chooseRun(key) {
  selectedRun = key; replayValue = null; selectedNode = null; selectedFinding = null; lastVersion = ""; stopPlayback();
  try { await loadSnapshot(); await loadRuns(); }
  catch(error) { toast(error.message); }
}
async function loadSnapshot() {
  if (!selectedRun) return;
  const key = selectedRun;
  const result = await api(`/api/runs/${encodeURIComponent(key)}`);
  if (selectedRun !== key) return;
  data = result;
  renderRetryNotice();
  const version = `${key}:${result.last_sequence}:${result.status}:${result.sources.length}`;
  if (version === lastVersion) return;
  lastVersion = version; render();
}
function metric(label, value, unit, fraction, note) {
  const bar = fraction == null ? "" : `<div class="microbar"><i style="width:${Math.max(0, Math.min(100, fraction * 100))}%"></i></div>`;
  return `<div class="metric"><div class="metric-label">${label}<span>${icon("external")}</span></div><div class="metric-value">${value}<small>${unit}</small></div>${bar}<div class="metric-note">${note}</div></div>`;
}
function providerError(reason, sequence = Infinity) {
  const connection = ["OpenAIConnectionError", "APIConnectionError"].includes(reason);
  const timeout = ["OpenAITimeoutError", "APITimeoutError"].includes(reason);
  if (!connection && !timeout) return "";
  const route = data.model_routes?.findLast(route => route.sequence <= sequence);
  const model = route?.model || data.model || "";
  const provider = model.startsWith("deepseek:") || model.startsWith("deepseek-") ? "DeepSeek" : "模型服务";
  return `${provider} ${connection ? "连接错误" : "请求超时"}`;
}
function renderRetryNotice() {
  const notice = $("#retry-notice");
  const node = data.status === "running" && data.model_retries?.findLast(r => r.end_sequence == null);
  notice.hidden = !node;
  if (!node) return;
  const retry = node.details.arguments;
  const reason = retry.http_status ? `DeepSeek HTTP ${retry.http_status}` : providerError(retry.error_type, node.sequence) || "DeepSeek 请求失败";
  const remaining = Math.max(0, Math.ceil((Date.parse(retry.time) + retry.delay_seconds * 1000 - Date.now()) / 1000));
  const progress = node.started_sequence || !remaining ? "正在重新连接" : `等待 ${remaining} 秒`;
  const message = `${reason} · 第 ${retry.retry_number}/${retry.max_retries} 次重试 · ${progress}`;
  if (notice.textContent !== message) notice.textContent = message;
}
function render() {
  const b = data.budget, lim = data.limits;
  const modelLimit = lim.max_model_calls === undefined ? 24 : lim.max_model_calls;
  const tokenLimit = lim.max_tokens === undefined ? 80000 : lim.max_tokens;
  const title=data.task.event || data.question;
  $("#question-title").textContent = title.length>65 ? title.slice(0,65)+"…" : title;
  $("#full-question").textContent=data.question;
  $("#condition-label").textContent = data.workflow === "adaptive-event-map" ? "事件调查" : "历史流程 · " + data.condition.toUpperCase();
  $("#run-badge").textContent = data.mock ? "OFFLINE FIXTURE" : "LIVE RESEARCH";
  $("#status").textContent = statusNames[data.status] || data.status; $("#status").className = `status ${data.status}`;
  const lastRoute = data.model_routes?.at(-1);
  const stageNames = {scope_planning:"范围规划",gap_planning:"缺口规划",retrieval_selection:"检索读取",evidence_assessment:"证据判断",evidence_repair:"证据纠错",report_draft:"报告初稿",report_review:"报告核对",temporal_review:"日期与分项审核"};
  const effortNames = {low:"低强度思考",high:"高强度思考",max:"最高强度思考"};
  const thinkingLabel = lastRoute?.thinking === "enabled"
    ? effortNames[lastRoute.reasoning_effort] || "思考模式" : "普通模式";
  const modelLabel = data.mock ? "合成事件 · 脚本模型工程验收" : lastRoute ? `${stageNames[lastRoute.stage] || lastRoute.stage} · ${lastRoute.model.replace("deepseek:", "")} · ${thinkingLabel}` : (data.model || "正在连接模型");
  $("#run-subtitle").textContent = `${modelLabel} / ${data.run_id} / ${data.task.mode || "retrospective"}`;
  const partialReasons = {
    cancelled_by_user: "本次调查已手动停止",
    research_budget_exhausted: "已为写作保留预算，仍有待补证问题",
    answer_binding_unavailable: "原文尚未形成有效结论绑定",
    repeated_unavailable_tool: "本轮因连续调用不可用工具而提前收尾，仍有分项未查完",
    adaptive_gaps_resolved_or_stalled: "本轮调查已收尾，请查看仍开放的缺口",
    adaptive_no_progress: "连续操作未获得新证据，已用现有证据收尾",
    current_check_incomplete: "已结束最新消息核查，现有一手证据不足以确认结果",
    source_only_report_after_draft_connection_failure: "报告初稿遇到连接错误，已用保存的原文完成复核并生成证据报告",
  };
  const noticeReason = data.stop_reason === "answer_binding_unavailable" && b.stop_reason === "adaptive_no_progress" ? b.stop_reason : data.stop_reason;
  const currentConclusionInReport = data.report_brief?.current && ["current_fact_checked", "current_check_incomplete"].includes(noticeReason);
  const notice = data.mock ? "此运行使用合成资料和脚本模拟模型，只验证工程流程，不是对真实问题的回答。" : data.status === "partial" && !currentConclusionInReport ? `调查部分完成：${partialReasons[noticeReason] || "本轮已停止，部分问题仍需核查"}。已有成果和缺口均已保留。` : data.status === "failed" || data.status === "interrupted" ? `${providerError(data.stop_reason) || "本次运行未正常完成"}。可检查节点结果和已保存证据。` : "";
  $("#notice").textContent = notice; $("#notice").hidden = !notice;
  $("#metrics").innerHTML = metric("模型请求", num(b.model_calls), modelLimit == null ? "无累计上限" : `上限 ${num(modelLimit)}`, modelLimit == null ? null : (b.model_calls || 0)/modelLimit, "包含核验与报告修订")
    + metric("Token 消耗", num(b.tokens), tokenLimit == null ? "无累计上限" : `/ ${num(tokenLimit)}`, tokenLimit == null ? null : (b.tokens || 0)/tokenLimit, `${b.unknown_usage_calls || 0} 次未知用量预留`)
    + metric("实际网页请求", num(b.retrieval_api_calls), "次", (b.retrieval_api_calls || 0)/24, `${b.web_search_requests || 0} 搜索 · ${b.web_extract_requests || 0} 提取`)
    + metric("已保存来源", num(data.sources.length), "篇", data.sources.length/8, `${b.retrieval_credits || 0} credits · ${b.retrieval_unknown_usage_calls || 0} 次未知 · 费用未定价`);
  $("#node-count").textContent = data.nodes.length; $("#timeline-count").textContent = data.timeline.length; $("#source-count").textContent = data.sources.length;
  $("#followup").disabled = false; $("#export-report").disabled = !data.report; $("#cancel-run").hidden = !data.can_cancel;
  $("#replay").max = data.last_sequence; $("#replay").value = replayValue ?? data.last_sequence;
  renderEvents(); renderGraph(); renderTimeline(); renderReport(); renderSources();
  if (selectedNode) inspectNode(selectedNode);
  const url = new URL(location.href); url.searchParams.set("run", selectedRun); history.replaceState(null, "", url);
}
function visibleNodes() {
  const cursor = replayValue ?? data.last_sequence;
  return data.nodes.filter(node => node.sequence <= cursor).map(node => {
    let copy = {...node};
    if (node.updates) {
      const update = node.updates.filter(u => u.sequence <= cursor).at(-1);
      if (update) copy = {...copy, ...update, details:update};
    } else if (node.end_sequence != null && node.end_sequence <= cursor) copy.status = node.final_status;
    else if (node.started_sequence != null && node.started_sequence <= cursor) copy.status = "running";
    else if (node.provenance === "actual_tool_call") copy.status = "running";
    if (copy.details?.kind === "run_finished") {
      if (copy.details.status === "failed") {
        copy.title = "运行中断，已保留证据";
        copy.status = "blocked";
      } else if (copy.details.reason === "source_only_report_after_draft_connection_failure") {
        copy.title = "交付证据报告";
      }
    }
    return copy;
  });
}
function renderGraph() {
  if (!data) return;
  const nodes = visibleNodes(), positions = new Map();
  $("#empty-graph").hidden = nodes.length > 0;
  $("#replay-label").textContent = replayValue == null ? "实时 / 最新" : `事件 ${replayValue} / ${data.last_sequence}`;
  const canvas = $("#graph-canvas"), h = Math.ceil(nodes.length/3)*151 + 35;
  canvas.style.height = `${h}px`;
  const width = $("#graph-viewport").clientWidth;
  const scale = Math.min(1, (width-15)/752);
  canvas.style.transform = `scale(${scale})`; canvas.style.marginBottom = `${h * (scale - 1)}px`;
  $("#graph-nodes").innerHTML = nodes.map((node, i) => {
    const x = (i%3)*244 + 18, y = Math.floor(i/3)*151 + 23;
    positions.set(node.id,{x:x+109,y:y+53});
    const isPublic = node.provenance === "model_reported";
    let label = isPublic ? "模型报告的计划状态" : statusNames[node.status] || node.status;
    if (isPublic && node.reason) label = node.reason;
    if (node.public_reason) label = node.public_reason;
    if (node.status === "blocked") label = providerError(node.details?.error_type || node.details?.reason, node.sequence) || node.details?.result?.error || node.details?.reason || label;
    if (node.label === "连接重试") {
      label = ({waiting:`等待 ${node.details.arguments.delay_seconds} 秒`,running:"正在重新连接",completed:"重试成功",blocked:"重试未成功"})[node.status] || label;
    }
    return `<button class="graph-node ${esc(node.provenance)} ${esc(node.status)} ${selectedNode === node.id ? "selected" : ""}" style="left:${x}px;top:${y}px" data-node="${esc(node.id)}" aria-label="${esc((node.label || kinds[node.kind]) + '：' + node.title)}"><span class="node-top"><span class="node-tag"><span>${icons[node.kind] || icon("point")}</span>${esc(node.label || kinds[node.kind] || node.kind)}</span><span>${String(i+1).padStart(2,"0")}</span></span><strong class="node-title">${esc(node.title)}</strong><span class="node-bottom">${esc(label).slice(0,80)}</span></button>`;
  }).join("");
  $("#graph-edges").setAttribute("viewBox",`0 0 752 ${h}`);
  $("#graph-edges").innerHTML = '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9eb296"/></marker></defs>' + data.edges.filter(edge => positions.has(edge.from) && positions.has(edge.to) && edge.sequence <= (replayValue ?? data.last_sequence)).map(edge => {
    const a = positions.get(edge.from), b = positions.get(edge.to);
    let d;
    if (a.y === b.y) d = `M ${a.x+109} ${a.y} L ${b.x-113} ${b.y}`;
    else d = `M ${a.x} ${a.y+55} C ${a.x} ${a.y+94},${b.x} ${b.y-92},${b.x} ${b.y-57}`;
    return `<path class="edge ${edge.kind}" d="${d}" marker-end="url(#arrow)"/>`;
  }).join("");
}
function inspectNode(id) {
  const node = visibleNodes().find(n => n.id === id);
  if (!node) {selectedNode=null;$("#inspector").innerHTML='<div class="inspector-label">REPLAY</div><h3>节点尚未出现</h3><p>回放到后续动作后，再查看它的观察结果。</p>';return;}
  selectedNode = id;
  const detail = node.details || {}, args = detail.arguments || detail;
  const payload = node.end_sequence > (replayValue ?? data.last_sequence) ? null : detail.result;
  let body = `<div class="inspector-label">${esc(node.label || kinds[node.kind] || "NODE")} / ${esc(node.provenance === "model_reported" ? "模型公开摘要" : "实际执行记录")}</div><h3>${esc(node.title)}</h3>`;
  if (node.provenance === "model_reported") body += `<p>${esc(node.reason || args.reason || "未附加公开决策说明。")}</p><p>模型报告状态：${esc(statusNames[node.status] || node.status)}</p>`;
  else body += `<p>${esc(statusNames[node.status] || node.status)}${node.time ? " · "+esc(new Date(node.time).toLocaleTimeString("zh-CN",{timeZone:"Asia/Hong_Kong",hour12:false}))+" HKT" : ""}</p>`;
  const serviceError = providerError(detail.error_type || detail.reason, node.sequence);
  if (serviceError) body += `<p>${esc(serviceError)}。下方保留 SDK 的原始异常名。</p>`;
  if(node.public_reason) body += `<h4>模型公开的搜索目标</h4><p>${esc(node.public_reason)}</p>`;
  if (node.url) body += `<p><a href="${esc(node.url)}" target="_blank" rel="noopener noreferrer">打开原始来源 ${icon("external")}</a></p>`;
  body += `<h4>${node.provenance === "model_reported" ? "计划与关联" : "实际参数"}</h4><pre>${esc(JSON.stringify(args,null,2))}</pre>`;
  if (payload) {
    let summary = payload;
    if (payload.results) summary = {results:payload.results};
    else if (payload.passages) summary = {version_id:payload.version_id,returned_passages:payload.passages.length,total_passages:payload.total_passages,next_offset:payload.next_offset};
    body += `<h4>观察结果</h4><pre>${esc(JSON.stringify(summary,null,2)).slice(0,6500)}</pre>`;
  }
  if (node.requests?.length) body += `<h4>服务请求</h4>${node.requests.filter(r => r.sequence <= (replayValue ?? data.last_sequence)).map(r=>`<p>${esc(r.operation)} · ${r.result && r.result.sequence <= (replayValue ?? data.last_sequence) ? (r.result.usage_known ? esc(r.result.credits)+" credits" : "用量未知") : "请求已派发"}</p>`).join("")}`;
  const pids = node.passage_ids || args.passage_ids || [];
  if (pids.length && payload !== null) body += `<h4>原文指针</h4>${pids.slice(0,8).map(citeButton).join(" ")}`;
  $("#inspector").innerHTML = body;
  document.querySelectorAll(".graph-node").forEach(n => n.classList.toggle("selected",n.dataset.node===id));
}
function citeButton(pid) {
  const sourceIndex = data.sources.findIndex(s => s.passages.some(p => p.passage_id === pid));
  return `<button class="citation" data-passage="${esc(pid)}" title="${esc(pid)}">${sourceIndex < 0 ? "未知引用" : "来源 "+String(sourceIndex+1).padStart(2,"0")} ${icon("external")}</button>`;
}
function inlineMarkdown(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\[(p-[a-f0-9]{16})\]/g,(_,pid)=>citeButton(pid)).replace(/\b(ef-[a-f0-9]{16})\b/g,(_,fid)=>`<button class="citation" data-finding="${fid}">地图节点 ↗</button>`);
}
function markdown(text) {
  let table = false, output = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("|")) {
      if (!table) {output.push("<table>");table=true;}
      const cells = line.replace(/^\||\|$/g,"").split("|");
      if (!cells.every(cell => /^[\s:\-]+$/.test(cell))) output.push("<tr>"+cells.map(c=>`<td>${inlineMarkdown(c.trim())}</td>`).join("")+"</tr>");
      continue;
    }
    if (table) {output.push("</table>");table=false;}
    const h = line.match(/^(#{1,4}) (.*)/);
    if (h) output.push(`<h3>${inlineMarkdown(h[2])}</h3>`);
    else if (line.trim()) output.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  if (table) output.push("</table>"); return output.join("");
}
function renderTimeline() {
  const ready = row => row.date_known !== false && row.date !== "日期未知" && (!row.temporal_verdict || row.temporal_verdict === "supported");
  const dated=data.timeline.filter(ready), pending=data.timeline.filter(row=>!ready(row));
  const cards = rows => rows.map(row => `<article class="milestone"><time>${esc(row.date_label || row.date)}</time><div class="milestone-card"><span class="small-label">${row.kind === "planned" ? "计划发生" : "来源记载 / 未独立核实"}</span><p>${inlineMarkdown(row.text.replace(/\[p-[a-f0-9]{16}\]/g,""))}</p><div>${row.citations.map(c=>citeButton(c.passage_id)).join(" ") || '<small class="muted">未关联到有效原文指针</small>'}</div></div></article>`).join("");
  $("#timeline-panel").innerHTML = `<div class="section-heading"><div><h2>把事件放回时间里</h2><p>${esc(data.timeline_notice)}</p></div><span class="small-label">${dated.length} 个日期节点 · ${pending.length} 条待确认或待复核</span></div><div class="timeline-list">${cards(dated) || '<p>尚未形成带日期依据的时间节点。</p>'}</div>${pending.length ? `<details><summary>日期待确认与待复核的相关证据</summary><p>保留原文，暂不放入有序时间线。</p><div class="timeline-list">${cards(pending)}</div></details>` : ""}`;
}
function renderReport() {
  if (data.report_brief_html) {
    const panel = $("#report-panel");
    const signature = data.run_id + ":" + data.last_sequence;
    if (panel.dataset.briefRun === signature) return;
    panel.dataset.briefRun = signature;
    panel.innerHTML = data.report_brief_html;
    return;
  }
  delete $("#report-panel").dataset.briefRun;
  const counts = new Map();
  for (const source of data.sources) {let host="未知域名"; try{host = new URL(source.url).hostname.replace(/^www\./,"");}catch{} counts.set(host,(counts.get(host)||0)+1);}
  const bars = [...counts].map(([host,count])=>`<div class="source-bar"><span>${esc(host)}<b>${count}</b></span><div class="microbar"><i style="width:${100*count/Math.max(1,data.sources.length)}%"></i></div></div>`).join("");
  const text = data.report.replace(/^停止原因：([a-zA-Z]+)。/m, (line, code) => {
    const message = providerError(code);
    return message ? `停止原因：${message}。` : line;
  });
  const reportStatus = data.report_revision?.output_mode === "event_report_source_only_recovery"
    ? "已根据原文生成报告" : data.model_finalized ? "模型已完成写作" : "尚未完成写作";
  $("#report-panel").innerHTML = `<div class="section-heading"><div><h2>从证据到回答</h2><p>点击引用可查看本次保存的原文。</p></div><span class="small-label">${esc(reportStatus)}</span></div><div class="report-layout"><article class="report-body">${text ? markdown(text) : '<div class="empty-panel">研究进行中。完成或停止后，报告将出现在这里。</div>'}</article><aside class="report-side"><h3>证据来源分布</h3>${bars || '<p>还没有读取来源。</p>'}<p>按保存的网页篇数统计，不是来源独立性或可信度评分。</p><h3>写作与核对</h3><p>${esc(data.report_revision?.status || "尚未进行报告修订")}</p><p>模型修订和原文引用都不能替代事实判断；未决问题需保留。</p></aside></div>`;
}
function renderSources() {
  const prior = $("#source-filter")?.value || "";
  const expanded = [...document.querySelectorAll('.source-card details[open]')].map(n=>n.closest('[data-source]').dataset.source);
  $("#sources-panel").innerHTML = `<div class="section-heading"><div><h2>每条证据，保留原样</h2><p>精确位置基于本次保存正文；实时提取内容并非已验证的历史快照。</p></div><div class="source-tools"><input id="source-filter" placeholder="查找来源或原文" aria-label="查找来源或原文" value="${esc(prior)}"></div></div><div class="source-list">${data.sources.map((s,i)=>`<article class="source-card" data-source="${esc(s.version_id)}"><header><span class="source-number">${String(i+1).padStart(2,"0")}</span><div><div class="source-title">${esc(s.title)}</div><small>${esc(s.url || "无公开网址")}</small><small>抓取 / 观察：${esc(s.retrieved_at ? new Date(s.retrieved_at).toLocaleString("zh-CN",{timeZone:"Asia/Hong_Kong",hour12:false})+" HKT" : "未记录")} · 历史可用：${esc(s.available_at || "未证明")}</small></div>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">原站 ${icon("external")}</a>` : ""}</header><details><summary>查看 ${s.passages.length} 个保存片段 · ${esc((s.content_sha256 || "").slice(0,12))}</summary>${s.passages.map(p=>`<section class="passage" id="${p.passage_id}"><small>${p.passage_id} · 原文位置 ${p.start}–${p.end}</small><p>${esc(p.text)}</p></section>`).join("")}</details></article>`).join("") || '<div class="empty-panel">尚未读取任何可保存的原文。</div>'}</div>`;
  for (const id of expanded) { const card=[...document.querySelectorAll('[data-source]')].find(n=>n.dataset.source===id); if(card) card.querySelector('details').open=true; }
  filterSources(prior);
}
function filterSources(text) { document.querySelectorAll(".source-card").forEach(card => {card.hidden = !card.textContent.toLowerCase().includes(text.toLowerCase());}); }
function setTab(name) {
  activeTab = name;
  document.querySelectorAll("[data-tab]").forEach(button => button.setAttribute("aria-selected",String(button.dataset.tab === name)));
  ["events","graph","timeline","report","sources"].forEach(tab => {$("#"+tab+"-panel").hidden = tab !== name;});
  if (name === "graph") renderGraph();
}
function showPassage(pid) {
  setTab("sources"); $("#source-filter").value="";filterSources("");
  const element = document.getElementById(pid);
  if (!element) return toast("没有找到这条引用对应的保存原文。");
  element.closest("details").open = true; element.classList.add("highlight"); element.scrollIntoView({behavior:"smooth",block:"center"});
}
function openDialog(followup=false) {
  parentRun = followup ? selectedRun : null;
  $("#dialog-title").textContent = followup ? "继续追问，或修改条件" : "你想了解哪件事？";
  $("#research-question").value = ""; $("#form-error").textContent="";
  $("#parent-note").hidden=!followup; $("#run-dialog").showModal(); $("#research-question").focus();
}
async function startRun(mode) {
  const button = $("#start-run"); button.disabled = true; $("#fixture-run").disabled=true; $("#form-error").textContent="";
  try {
    const request = {question:$("#research-question").value + ($("#target-date").value ? "\n调查截止日期：" + $("#target-date").value : ""), workflow:$("#workflow").value, model_profile:mode === "fixture" ? "single" : $("#model-profile").value, condition:$("#condition").value, mode, parent_run_id:mode === "fixture" ? null : parentRun};
    const result = await api("/api/runs",{method:"POST",body:JSON.stringify(request)});
    $("#run-dialog").close(); await chooseRun(result.id); setTab("graph"); toast(mode === "fixture" ? "正在运行合成资料与脚本模型验收，不调用付费 API。" : "调查已开始，Agent 将自行搜索与选择网页。");
  } catch(error) { $("#form-error").textContent = error.message; }
  finally {button.disabled=false;$("#fixture-run").disabled=false;}
}
function stopPlayback() { clearInterval(playback);playback=null;$("#play-replay").innerHTML=icon("play"); }
$("#workflow").addEventListener("change",()=>{$("#condition").disabled=$("#workflow").value!=="legacy";if($("#condition").disabled)$("#condition").value="baseline";});
$("#new-run").addEventListener("click",()=>openDialog());
$("#followup").addEventListener("click",()=>openDialog(true));
$("#close-dialog").addEventListener("click",()=>$("#run-dialog").close());
$("#open-help").addEventListener("click",()=>$("#help-dialog").showModal());
$("#close-help").addEventListener("click",()=>$("#help-dialog").close());
$("#refresh-runs").addEventListener("click",()=>loadRuns().catch(e=>toast(e.message)));
$("#run-form").addEventListener("submit",e=>{e.preventDefault();startRun("live");});
$("#fixture-run").addEventListener("click",()=>startRun("fixture"));
$("#replay").addEventListener("input",e=>{stopPlayback();replayValue=Number(e.target.value);renderGraph();if(selectedNode)inspectNode(selectedNode);});
$("#live-view").addEventListener("click",()=>{stopPlayback();replayValue=null;$("#replay").value=data?.last_sequence||0;renderGraph();});
$("#play-replay").addEventListener("click",()=>{
  if (!data) return;if(playback)return stopPlayback();
  replayValue = replayValue == null || replayValue >= data.last_sequence ? 0 : replayValue;
  $("#play-replay").innerHTML=icon("pause");
  playback=setInterval(()=>{const next=data.nodes.map(n=>n.sequence).filter(s=>s>replayValue).sort((a,b)=>a-b)[0];if(next==null){stopPlayback();replayValue=null;}else replayValue=next;$("#replay").value=replayValue??data.last_sequence;renderGraph();},700);
});
$("#fit-graph").addEventListener("click",()=>{renderGraph();$("#graph-viewport").scrollTo({top:0,left:0,behavior:"smooth"});});
$("#export-report").addEventListener("click",()=>{if(selectedRun)location.assign(`/api/runs/${encodeURIComponent(selectedRun)}/report.html`);});
$("#cancel-run").addEventListener("click",async()=>{try{await api(`/api/runs/${encodeURIComponent(selectedRun)}/cancel`,{method:"POST",body:"{}"});toast("已请求停止，将保留已读取证据、用量与部分报告。");}catch(e){toast(e.message);}});
document.addEventListener("click",e=>{const f=e.target.closest("[data-finding]");if(f){setTab("events");inspectFinding(f.dataset.finding);}const a=e.target.closest("[data-action]");if(a){setTab("graph");replayValue=null;renderGraph();inspectNode("tool:"+a.dataset.action);}const all=e.target.closest("[data-expand-minor]");if(all){$("#events-panel").classList.toggle("show-minor");all.setAttribute("aria-expanded",String($("#events-panel").classList.contains("show-minor")));}const node=e.target.closest("[data-node]");if(node)inspectNode(node.dataset.node);const run=e.target.closest("[data-run]");if(run)chooseRun(run.dataset.run);const tab=e.target.closest("[data-tab]");if(tab)setTab(tab.dataset.tab);const citation=e.target.closest("[data-passage]");if(citation){e.preventDefault();showPassage(citation.dataset.passage);}});
document.addEventListener("input",e=>{if(e.target.id==="source-filter")filterSources(e.target.value);});
window.addEventListener("resize",()=>{if(activeTab==="graph")renderGraph();});
async function poll() {if(pollBusy)return;pollBusy=true;try{await loadSnapshot();await loadRuns();}catch(error){$("#connection").textContent="连接已中断 · 检查本地服务";}finally{pollBusy=false;}}
health().then(()=>loadRuns(true)).catch(e=>{toast(e.message);$("#connection").textContent="服务未连接";});
setInterval(poll,1800);

const findingNames = {observed:"来源记载已发生",planned:"计划 / 尚未等同发生",statement:"来源陈述",unknown:"发生类型未知"};
const evidenceStates = {source_supported:"有原文支持",source_context:"部分证据",important_unknown:"重要未知",hypothesis:"解释假设",disputed:"存在争议",changes_over_time:"状态随时间变化",corrected:"已有更正",historical_state:"历史状态",counterevidence_recorded:"记录了反证"};
const gapNames = {temporal_audit:"时间或适用范围待补证",question_or_explanation_unresolved:"问题尚未完整回答",source_check:"出处与独立性待核查",missing_original:"尚缺原文",missing_event_date:"缺事件日期依据",chronology_unanchored:"时间脉络尚未建立",conflicting_evidence:"存在冲突证据"};
const gapTitle = gap => gap.reference_month ? `${gap.reference_month} 数据待补证` : gap.question_part_id ? gap.title : gapNames[gap.kind] || gap.kind;
const seenFindings = new Set();
let selectedFinding = null;
let lastAtlasSignature = "";
function eventDate(span) {return span?.start ? span.start + (span.end && span.end!==span.start ? " → " + span.end : "") : "日期未明确";}
function renderEvents() {
  const view=data.event_view || {}, units=view.event_units || [], nodes=view.nodes || [];
  const signature=JSON.stringify([data.run_id,view,data.gaps,data.status,data.nodes.at(-1)]);
  if(signature===lastAtlasSignature)return;
  lastAtlasSignature=signature;
  $("#event-count").textContent=units.length;
  if(!view.available){$("#events-panel").innerHTML=`<div class="empty-panel">${esc(view.error || (data.workflow==="legacy" ? "这份历史记录使用旧流程，没有事件地图。可查看调查过程、时间线和报告。" : "调查范围与事件节点建立后，会在这里逐步出现。"))}</div>`;return;}
  const lastAction = [...data.nodes].reverse().find(n=>n.provenance==="actual_tool_call");
  const ribbon=data.status==="running" ? `<div class="live-ribbon">${icon("flame")}<span>正在调查</span><strong>${esc(lastAction?.title || "准备调查范围")}</strong></div>` : `<div class="live-ribbon settled">${icon("flame")}<span>已保存调查</span><strong>${esc(statusNames[data.status]||data.status)} · ${units.length} 条原文发现</strong></div>`;
  const panels=nodes.map(node=>{
    const findings=units.filter(u=>u.question_node_id===node.node_id).sort((a,b)=>(a.finding.event_time.start || "9999").localeCompare(b.finding.event_time.start || "9999"));
    const gaps=(data.gaps||[]).filter(g=>g.node_id===node.node_id && g.status!=="resolved");
    return `<section class="event-branch ${node.importance==="minor" ? "minor-branch" : ""}"><div class="branch-root"><span class="branch-flame">${icon("flame")}</span><div><small>${node.importance==="major"?"关键问题":"补充节点"} · ${esc(evidenceStates[node.status]||node.status)}</small><h3>${esc(node.title)}</h3><p>${esc(node.importance_reason||"")}</p></div></div><div class="question-parts">${(node.part_status||[]).map(p=>`<span class="part ${p.status==="model_assessed_answered"?"covered":"open"}">${esc(p.question)} · ${p.status==="model_assessed_answered"?"有对应原文":p.status==="disputed"?"争议":"待补证"}</span>`).join("")}</div><div class="event-spine">${findings.map(u=>{
      const fresh=!seenFindings.has(data.run_id+u.event_unit_id);seenFindings.add(data.run_id+u.event_unit_id);
      return `<button class="event-card ${fresh?"arrive":""} ${u.finding.stance==="contradicts"?"opposed":""}" data-finding="${esc(u.event_unit_id)}"><span class="event-pin">${icon("flame")}</span><time>${esc(eventDate(u.finding.event_time))}</time><small>${esc(findingNames[u.finding.occurrence])}${u.finding.stance==="contradicts"?" · 相反证据":""}</small><p>${esc(u.finding.assertion)}</p><span class="event-card-foot">${new Set(u.references.map(r=>r.version_id)).size} 个来源记录 · 查看依据 ${icon("arrow")}</span></button>`;
    }).join("") || '<p class="unknown-card">尚无成功写回的原文发现；已经搜索或读取不代表问题已回答。</p>'}</div>${gaps.map(g=>`<div class="gap-leaf">${icon("unknown")}<span>${esc(gapTitle(g))} · ${g.status==="stalled"?"暂时停滞":"仍开放"} · 已检索 ${g.attempts||0} 轮</span></div>`).join("")}</section>`;
  }).join("");
  const relations=(view.relations||[]).map(r=>`<p>${esc(nodes.find(n=>n.node_id===r.first)?.title||r.first)} → ${esc(r.relation)} → ${esc(nodes.find(n=>n.node_id===r.second)?.title||r.second)}<br><small>${esc(r.reason)} · 模型提出的关系，按引用判断</small></p>`).join("");
  $("#events-panel").innerHTML=`<div class="section-heading"><div><small class="small-label">EVENT ATLAS</small><h2>事件的脉络，证据的来路</h2><p>每个节点是一条保存的发现。点击展开依据、未知和对应调查动作。</p></div><button class="quiet-button" data-expand-minor aria-expanded="false">展开次要节点</button></div>${ribbon}${(data.gaps||[]).filter(g=>!g.node_id&&g.status!=="resolved").map(g=>`<p class="gap-leaf">${esc(gapTitle(g))} · ${esc(g.status)}</p>`).join("")}<div class="atlas-layout"><div class="event-atlas">${panels}${relations?`<section class="event-relations"><h3>已记录的事件关系</h3>${relations}</section>`:""}</div><aside id="event-inspector" class="inspector"><div class="inspector-label">EVIDENCE / CONTEXT</div><h3>选择一簇火苗</h3><p>查看这条发现的原文、事件日期、来源说明与实际写入动作。</p><p>地图显示当前累计状态；调查过程页可回放动作顺序。</p></aside></div><p class="footnote">${esc(view.notice)} · 不同网页不自动等于独立来源；相邻节点不自动构成因果。</p>`;
  const expandButton=document.querySelector("[data-expand-minor]");
  if(expandButton)expandButton.setAttribute("aria-expanded",String($("#events-panel").classList.contains("show-minor")));
  if(selectedFinding && units.some(u=>u.event_unit_id===selectedFinding))inspectFinding(selectedFinding);
}
function inspectFinding(id) {
  const unit=data.event_view?.event_units.find(u=>u.event_unit_id===id);if(!unit)return;
  selectedFinding=id;
  const f=unit.finding;
  $("#event-inspector").innerHTML=`<div class="inspector-label">SAVED FINDING</div><h3>${esc(f.assertion)}</h3><p>${esc(findingNames[f.occurrence])} · ${esc(f.stance)}</p><h4>事件日期</h4><p>${esc(unit.time_labels?.event || eventDate(f.event_time))}</p><h4>结论适用时段</h4><p>${esc(unit.time_labels?.applicability || (f.valid_time?.start ? eventDate(f.valid_time) : (unit.time_labels?.applicability_note || "未单独记录适用时段；不代表事件日期未知或结论一直有效。")))}</p><h4>时间审核</h4><p>${esc(unit.temporal_review?.event_reason || "尚未完成单独时间审核；日期为原记录。")}</p>${(unit.temporal_repairs||[]).map(t=>`<p>${esc(t.kind === "event_date_basis" ? "事件日期补证" : "适用范围补证")}：${esc({resolved:"补充发现已通过复审",retained_unknown:"补证后仍未知",open:"待补证",awaiting_review:"待复审"}[t.status]||t.status)}${t.resolved_by ? ` <button class="quiet-button" data-finding="${esc(t.resolved_by)}">查看补充发现 ↗</button>` : ""}${t.outcome_reason ? `<br>${esc(t.outcome_reason)}` : ""}</p>`).join("")}<h4>不确定性</h4><p>${esc(f.uncertainty||"记录未附加不确定性说明；不代表事实认证。")}</p><h4>原文依据</h4>${unit.references.map(r=>`<blockquote>${esc(r.quote)}</blockquote>${citeButton(r.passage_id)}`).join("")}<h4>来源角色与页面日期</h4>${(unit.source_context||[]).map(c=>`<p>${esc(c.hosting_domain)} · 原文署名：${esc(c.attributed_origins?.map(o=>o.name).join("、")||"未明确识别")}<br>${esc(c.page_dates?.map(d=>d.role+"："+d.value).join("；")||"所选原文未明确页面日期")}</p>`).join("")}<p>承载域名与原始发布者分开；页面日期不能自动补为事件日期。</p><h4>实际动作</h4>${unit.action_id?`<button class="quiet-button" data-action="${esc(unit.action_id)}">查看证据写回步骤 ${icon("arrow")}</button>`:'<p>旧记录未保存动作关联。</p>'}`;
  document.querySelectorAll(".event-card").forEach(n=>n.classList.toggle("selected",n.dataset.finding===id));
}
