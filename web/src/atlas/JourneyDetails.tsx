import { useLayoutEffect, useRef, useState } from 'react';
import type { ResearchState, ResearchTask, TimelineView } from '@/research/state';
import { JourneyGlyph } from './Journey';
import { activityDetail, activityLabel, branchTitle, JourneyNode } from './journey-model';
import { linkedMarkdown, Report, webUrl } from './report';
import { useLocale } from './locale';

type Evidence = { url: string; title?: string; evidence_quote?: string };
const taskStatus = { pending: '等待中', running: '取證中', completed: '已交回', failed: '取證失敗', cancelled: '已停止' } as const;
const nodeStatus = { pending: '等待中', active: '進行中', done: '已完成', stopped: '已停止' } as const;

function taskEvidence(task: ResearchTask) {
  const sources = new Map<string, Evidence>();
  function add(source: Evidence) {
    const url = webUrl(source.url);
    if (!url) return;
    const key = new URL(url);
    key.hash = '';
    if (!sources.has(key.href)) sources.set(key.href, { ...source, url });
  }
  for (const source of (task.result?.evidence_records || []) as Evidence[]) add(source);
  for (const source of Object.values(task.result?.references || {})) add(source);
  return Array.from(sources.values());
}

function headings(markdown: string) {
  return Array.from(markdown.matchAll(/^#{1,3}\s+(.+?)\s*$/gm), match => match[1].replace(/[*_]/g, ''));
}

function openingFinding(markdown: string) {
  const blocks = markdown.replace(/\r/g, '').split(/\n\s*\n/);
  const summary = blocks.findIndex(text => /^#{1,6}\s+.*(?:核心(?:結論|结论|判定)|主要發現|主要发现|summary|findings)/i.test(text.trim()));
  const body = blocks.slice(summary + 1).map(block => block.trim()).find(text =>
    text && !/^(?:#|\x60\x60\x60|[-*_]{3,}$)/.test(text)
    && !/^(?:Based on my research|I have gathered|Here is my)/i.test(text));
  const text = body?.replace(/\[\[\d+\]\]/g, '').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_\x60]/g, '').replace(/\s+/g, ' ').trim();
  if (!text || text.length <= 245) return text;
  const end = Math.max(text.lastIndexOf('。', 245), text.lastIndexOf('；', 245), text.lastIndexOf('. ', 245));
  return end > 80 ? text.slice(0, end + 1) : text.slice(0, 245) + '…';
}

function Branches({ tasks, select, showFindings }: {
  tasks: ResearchTask[]; select: (id: string) => void; showFindings?: boolean;
}) {
  const t = useLocale();
  return <ul className="detail-branches">{tasks.map(task => {
    const finding = showFindings && task.result?.answer && openingFinding(task.result.answer);
    return <li key={task.id}><button type="button" onClick={() => select(task.id)}>
      <span><small className={`detail-task-status ${task.status}`}>{t(taskStatus[task.status])}</small>
        <strong>{branchTitle(task)}</strong></span><span aria-hidden="true">›</span>
    </button>{finding && <p>{finding}</p>}</li>;
  })}</ul>;
}

function BranchDetails({ task }: { task: ResearchTask }) {
  const t = useLocale();
  const [fullRecord, setFullRecord] = useState(false);
  const [moreSources, setMoreSources] = useState(false);
  const sources = taskEvidence(task);
  const finding = task.result?.answer && openingFinding(task.result.answer);
  return <>
    {finding && <section className="detail-section"><h4>{t('查證發現')}</h4><p className="detail-lead">{finding}</p></section>}
    <details className="detail-disclosure" open={!finding || undefined}>
      <summary>{t('查看完整查證問題')}</summary><p>{task.question}</p>
    </details>
    {!!task.steps?.length && <details className="detail-disclosure">
      <summary>{t('追查過程 · {n} 步', { n: task.steps.length })}</summary>
      <ol className="detail-periods">{task.steps.map((step, index) => <li key={index}>
        <small>{t(step.action === 'search' ? '檢索 · {n} 個結果' : '閱讀 · {n} 份原文', { n: step.count })}</small>
        <strong>{step.action === 'search' ? step.query : step.focus}</strong>
        {step.reason && <p>{step.reason}</p>}
      </li>)}</ol>
    </details>}
    {task.handoff && <section className="detail-section">
      <h4>{t('收束原因')}</h4>
      <p>{task.handoff.kind === 'loop_guard' ? t('相同操作未取得新線索，已交回現有發現。') : task.handoff.reason}</p>
      {task.handoff.gap && <><h4>{t('仍待查證')}</h4><p>{task.handoff.gap}</p></>}
    </section>}
    {task.result?.error && <p className="detail-error">{task.result.error}</p>}
    {task.result?.answer && <details className="detail-disclosure" onToggle={e => setFullRecord(e.currentTarget.open)}>
      <summary>{t('查看完整查證記錄')}</summary>
      {fullRecord && <article><Report article={linkedMarkdown(task.result.answer, task.result.references || {})}/></article>}
    </details>}
    {!!sources.length && <details className="detail-disclosure" open>
      <summary>{t('原文來源 · {n}', { n: sources.length })}</summary>
      <ul className="detail-sources">{sources.slice(0, 4).map(source => <EvidenceLink key={source.url} source={source}/>)}</ul>
      {sources.length > 4 && <details className="detail-disclosure detail-more" onToggle={e => setMoreSources(e.currentTarget.open)}>
        <summary>{t('查看更多來源 · {n}', { n: sources.length - 4 })}</summary>
        {moreSources && <ul className="detail-sources">{sources.slice(4).map(source => <EvidenceLink key={source.url} source={source}/>)}</ul>}
      </details>}
    </details>}
  </>;
}

function EvidenceLink({ source }: { source: Evidence }) {
  return <li>
    <a href={source.url} target="_blank" rel="noreferrer">{source.title || new URL(source.url).hostname} ↗</a>
    {source.evidence_quote && <p>{source.evidence_quote}</p>}
  </li>;
}

function TimelinePreview({ timeline, raw, seeTimeline }: {
  timeline?: TimelineView; raw?: ResearchState['timeline']; seeTimeline: () => void;
}) {
  const t = useLocale();
  const groups = timeline?.groups || [];
  const dated = raw?.filter(row => row.date !== 'unknown').sort((a, b) => a.date.localeCompare(b.date)) || [];
  return <>
    {!!groups.length && <ol className="detail-periods">{groups.map((group, index) => <li key={`${group.start}-${index}`}>
      <small>{group.start}{group.start !== group.end ? ` — ${group.end}` : ''}</small>
      <strong>{group.title}</strong>
      {group.summary && <p>{group.summary}</p>}
    </li>)}</ol>}
    {!groups.length && !!dated.length && <ol className="detail-periods">{dated.slice(0, 3).map((row, index) => <li key={index}>
      <small>{row.date}</small><p>{row.event}</p></li>)}</ol>}
    {(timeline || raw) && <button type="button" className="quiet detail-action" onClick={seeTimeline}>{t('查看完整時間線')} ↗</button>}
  </>;
}

function NodeContent({ node, state, timeline, select, seeReport, seeTimeline }: {
  node: JourneyNode; state: ResearchState; timeline?: TimelineView;
  select: (id: string) => void; seeReport: () => void; seeTimeline: () => void;
}) {
  const t = useLocale();
  const tasks = node.tasks || [];
  switch (node.kind) {
    case 'question':
      return state.question ? <p className="detail-question">{state.question}</p>
        : <p className="muted">輸入問題後，這裡會顯示本輪調查的原始提問。</p>;
    case 'plan':
      return tasks.length ? <section className="detail-section"><h4>{t('首輪查證方向 · {n}', { n: tasks.length })}</h4>
        <Branches tasks={tasks} select={select}/>
      </section> : <p className="muted">{node.status === 'active' ? t('正在拆解問題，查證方向建立後會顯示在這裡。') : t('本輪沒有已保存的查證方向。')}</p>;
    case 'dispatch':
      return <section className="detail-section"><h4>{node.preview}</h4><Branches tasks={tasks} select={select}/></section>;
    case 'review':
      return <section className="detail-section"><h4>{t('本輪分支與發現')}</h4>
        <Branches tasks={tasks} select={select} showFindings/>
      </section>;
    case 'branch':
      return node.task ? <BranchDetails task={node.task}/> : null;
    case 'time':
      return <TimelinePreview timeline={timeline} raw={state.timeline} seeTimeline={seeTimeline}/>;
    case 'writing':
    case 'delivery': {
      const article = state.article || state.partialArticle || '';
      const sections = headings(article);
      return <>
        {!!sections.length && <section className="detail-section"><h4>{state.article ? t('報告章節') : t('目前已寫章節')}</h4>
          <ul className="detail-sections">{sections.map((title, index) => <li key={index}>{title}</li>)}</ul>
        </section>}
        {article && <button type="button" className="quiet detail-action" onClick={seeReport}>
          {state.article ? t('查看完整報告') + ' ↗' : t('查看報告草稿') + ' ↗'}
        </button>}
      </>;
    }
  }
}

export function JourneyDetails({ open, node, state, timeline, select, seeReport, seeTimeline }: {
  open: boolean; node?: JourneyNode; state: ResearchState; timeline?: TimelineView;
  select: (id: string) => void; seeReport: () => void; seeTimeline: () => void;
}) {
  const t = useLocale();
  const scroll = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  useLayoutEffect(() => {
    if (scroll.current) scroll.current.scrollTop = 0;
    if (open && (window.matchMedia('(max-width: 690px)').matches || document.activeElement?.matches(':focus-visible'))) {
      heading.current?.focus({ preventScroll: true });
    }
  }, [node?.id, open]);
  const activity = node?.status === 'active' ? activityLabel(node.activity, t) : '';
  const detail = activity ? activityDetail(node?.activity, t) : '';
  return <aside id="journey-details" className={`journey-details${open ? ' is-open' : ''}`} aria-hidden={!open} aria-label={t('節點詳情')}>
    <header className="journey-details-top">
      <h3 ref={heading} tabIndex={-1}>{t('節點詳情')}</h3>
    </header>
    <div className="journey-details-scroll" ref={scroll}>
      {node ? <div className="detail-content" key={node.id}>
        <div className="detail-heading">
          <span className="graph-glyph"><JourneyGlyph kind={node.kind}/></span>
          <span className={`detail-status ${node.status}`}>{t(node.task ? taskStatus[node.task.status] : nodeStatus[node.status])}</span>
        </div>
        <h4 className="detail-title">{node.title}</h4>
        {activity && <div className="detail-activity"><strong>{activity}</strong>{detail && <p>{detail}</p>}</div>}
        <NodeContent node={node} state={state} timeline={timeline} select={select}
          seeReport={seeReport} seeTimeline={seeTimeline}/>
      </div> : <p className="detail-empty">{t('選擇線索圖上的節點查看內容。')}</p>}
    </div>
  </aside>;
}
