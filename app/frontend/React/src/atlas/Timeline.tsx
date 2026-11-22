import { useMemo } from 'react';
import { ResearchState, TimelineObservation, TimelineView } from '@/research/state';
import { webUrl } from './report';
import { useLocale } from './locale';
import type { MessageKey, Translate } from './i18n';

const roles: Record<string, MessageKey> = {
  observed: '已發生', announced: '已公告', planned: '計畫／推估', published: '來源記錄', uncertain: '待確認',
};

function badge(row: TimelineObservation, t: Translate) {
  if (row.date_basis === 'documented_by') return t('截至此日已有記錄');
  if (row.date === 'unknown' && row.role === 'observed') return t('有記錄，日期待查');
  return t(roles[row.role] || '事件記錄');
}

function visualRole(row: TimelineObservation) {
  return row.date_basis === 'documented_by' || row.date_basis === 'source_publication' ? 'published' : row.role;
}

function dateLabel(row: TimelineObservation, t: Translate) {
  if (row.date === 'unknown') return t('尚無日期依據');
  const range = row.date_end ? `${row.date} — ${row.date_end}` : row.date;
  if (row.date_basis === 'approximate') return t('約 {date}', { date: range });
  if (row.date_basis === 'documented_by') return t('截至 {date}', { date: range });
  if (row.date_basis === 'source_publication') return `${range} · ${t('來源日期')}`;
  return range;
}

function EvidenceRow({ row }: { row: TimelineObservation }) {
  const t = useLocale();
  return <details className={`event-row role-${visualRole(row)}`}>
    <summary>
      <div className="event-meta"><span>{dateLabel(row, t)}</span><span className="event-role">{badge(row, t)}</span></div>
      <p>{row.event}</p>
    </summary>
    <div className="event-evidence">
      {row.evidence.map((item, i) => <blockquote key={i}>
        <a href={webUrl(item.source_url) || undefined} target="_blank" rel="noreferrer">{t('原文')} ↗</a>
        {item.source_date && <small>{t('來源日期')} {item.source_date}</small> }
        <p>{item.quote}</p>
      </blockquote>)}
    </div>
  </details>;
}

function basicView(rows: TimelineObservation[], t: Translate): TimelineView {
  const dated = rows.filter(row => row.date !== 'unknown').sort((a, b) => a.date.localeCompare(b.date));
  return {
    groups: dated.length ? [{title:t('已整理的事件記錄'), start:dated[0].date, end:dated[dated.length - 1].date,
      events:dated}] : [],
    undated: rows.filter(row => row.date === 'unknown'),
  };
}

export function Timeline({ state, regroup, busy, error }: {
  state: ResearchState; regroup?: () => void; busy: boolean; error?: string;
}) {
  const t = useLocale();
  const view = useMemo(() => state.timelineView || basicView(state.timeline || [], t), [state.timelineView, state.timeline, t]);
  return <section className="report-panel timeline-panel">
    <header className="timeline-header">
      <div><h3>{t('事件時間線')}</h3><small>{t('由早到晚 · 展開里程碑可查看原文')}</small></div>
      {regroup && !state.timelineView && <button className="quiet" type="button" onClick={regroup} disabled={busy}>
        {busy ? t('正在整理…') : t('整理關鍵時段')}</button>}
    </header>
    <div className="timeline-key" aria-label={t('日期類別')}>
      {Object.entries(roles).map(([role, label]) => <span key={role} className={`role-${role}`}>{t(label)}</span>)}
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    {!view.groups.length && !view.undated.length && <p className="muted">{t('目前沒有可列出的時間證據。')}</p>}
    <ol className="timeline-groups">
      {view.groups.map((group, index) => <li className="timeline-group" key={`${group.start}-${index}`}>
        <div className="period-card">
          <div className="timeline-range"><span>{group.start}{group.start !== group.end ? ` — ${group.end}` : ''}</span>
            <h4>{group.title}</h4></div>
          {group.summary && <p className="period-summary">{group.summary}</p>}
          <div className="period-roles">{Array.from(new Set(group.events.map(visualRole))).map(role =>
            <span key={role} className={`event-role role-${role}`}>{t(roles[role] || '事件記錄')}</span>)}</div>
          <details className="period-evidence">
            <summary>{t('查看這段的證據 · {n} 組', { n: group.events.length })}</summary>
            {group.events.map((row, i) => <EvidenceRow row={row} key={i}/>)}
          </details>
        </div>
      </li>)}
    </ol>
    {!view.groups.length && !!view.undated.length && <p className="muted">{t('目前的資料尚未提供可用於排序的日期。')}</p>}
  </section>;
}
