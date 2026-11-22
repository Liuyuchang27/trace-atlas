import { FormEvent, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import useResearchSession from '@/hooks/useResearchSession';
import { ResearchSession, TimelineView, restoreSavedResearch } from '@/research/state';
import { linkedMarkdown, Report } from './report';
import { readSession } from '@/research/history';
import { migrateBrowserStorage } from '@/research/storage';
import { ResearchJourney } from './Journey';
import { JourneyDetails } from './JourneyDetails';
import { activityLabel, buildJourney } from './journey-model';
import { LanguageMenu } from './LanguageMenu';
import { Timeline } from './Timeline';
import { getLocale, Locale, translator } from './i18n';
import { LocaleContext, useLocale } from './locale';
import './style.css';

const labels = { idle: '等待問題', running: '調查中', completed: '研究完成', failed: '調查失敗', cancelled: '已停止' } as const;


function download(text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'trace-atlas-report.md';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Atlas() {
  const [locale, setLocale] = useState(getLocale);
  useEffect(() => {
    localStorage.setItem('trace_atlas_locale', locale);
    localStorage.setItem('trace_atlas_lang', locale);
    document.documentElement.lang = locale;
    document.title = `Trace Atlas · ${translator(locale)('研究工作台')}`;
  }, [locale]);
  return <LocaleContext.Provider value={translator(locale)}><Workbench locale={locale} setLocale={setLocale}/></LocaleContext.Provider>;
}

function Workbench({ locale, setLocale }: { locale: Locale; setLocale: (value: Locale) => void }) {
  const t = useLocale();
  const historyDate = new Intl.DateTimeFormat(locale, { timeZone: 'Asia/Hong_Kong', month: 'short', day: 'numeric' });
  const research = useResearchSession();
  const current = research.session.current;
  const [preview, setPreview] = useState<{ id: string; session: ResearchSession }>();
  const [round, setRound] = useState<number>();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const selection = useRef(0);
  const shown = preview?.session || research.session;
  const state = round === undefined ? shown.current : shown.rounds[round] || shown.current;
  const viewingHistory = !!preview || round !== undefined;
  const [question, setQuestion] = useState('');
  const [choice, setChoice] = useState('');
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [view, setView] = useState<'map' | 'report' | 'timeline'>('map');
  const [timeline, setTimeline] = useState(false);
  const [digestByRun, setDigestByRun] = useState<Record<string, TimelineView>>({});
  const [digestRequest, setDigestRequest] = useState<{runId: string; error?: string}>();
  const article = linkedMarkdown(state.article || state.partialArticle || '', state.references);
  const currentLabel = state.status === 'running' ? activityLabel(state.activity, t) || t(labels.running)
    : state.article.startsWith('⚠️') ? t('已交付現有發現') : t(labels[state.status]);
  const runId = state.recovery?.run_id;
  const digest = state.timelineView || (runId ? digestByRun[runId] : undefined);
  const journey = buildJourney({ ...state, timelineView: digest }, t);
  const selectedNode = journey.rows.flat().find(node => node.id === choice);

  const composerFirst = !state.question || viewingHistory || current.status === 'running';
  const detailsLabel = t(detailsOpen ? '收起節點詳情' : '展開節點詳情');

  function closeDetails() {
    setDetailsOpen(false);
    document.getElementById('journey-details-toggle')?.focus({ preventScroll: true });
  }

  function selectNode(id: string) {
    setChoice(id);
    setDetailsOpen(true);
  }

  function showCurrent() {
    selection.current += 1;
    setPreview(undefined);
    setRound(undefined);
    setHistoryError('');
    setChoice('');
    setDetailsOpen(false);
    setView('map');
  }

  async function openHistory(id: string) {
    const token = ++selection.current;
    setHistoryError('');
    setHistoryOpen(false);
    setRound(undefined);
    setChoice('');
    setDetailsOpen(false);
    setView('map');
    if (id === research.currentId) {
      setPreview(undefined);
      return;
    }
    if (current.status !== 'running') {
      setPreview(undefined);
      await research.selectConversation(id);
      return;
    }
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error();
      const saved = await response.json();
      const session = readSession(saved.data);
      const savedRun = session.current.recovery?.run_id;
      if (savedRun) {
        const stored = await fetch(`/api/research/${encodeURIComponent(savedRun)}`);
        if (stored.ok) session.current = restoreSavedResearch(session.current, await stored.json());
      }
      if (token === selection.current) setPreview({ id, session });
    } catch {
      if (token === selection.current) setHistoryError(t('暫時無法讀取這次調查。'));
    }
  }

  async function regroup() {
    if (!runId) return;
    setDigestRequest({ runId });
    try {
      const response = await fetch(`/api/research/${encodeURIComponent(runId)}/timeline-view`, { method: 'POST' });
      if (!response.ok) throw new Error();
      const result: TimelineView = await response.json();
      setDigestByRun(previous => ({ ...previous, [runId]: result }));
      setDigestRequest(previous => previous?.runId === runId ? undefined : previous);
    } catch {
      setDigestRequest(previous => previous?.runId === runId
        ? { runId, error: t('暫時無法整理時段，已儲存的事件仍可閱讀。') } : previous);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || current.status === 'running') return;
    showCurrent();
    void research.start(text, undefined, timeline, locale);
  }

  const composer = <section className="intro">
    {viewingHistory ? <div className="history-preview">
      <p>{preview ? t('正在查看歷史調查，原調查繼續執行。') : t('正在查看這次對話的較早一輪。')}</p>
      <button type="button" className="quiet" onClick={showCurrent}>{t('返回目前調查')}</button>
    </div> : <form onSubmit={submit}>
      <label htmlFor="question">{state.question ? t('繼續提問') : t('你想查清什麼？')}</label>
      <textarea id="question" placeholder={t('輸入研究問題，或接著追問目前的調查')}
        value={question} onChange={e => setQuestion(e.target.value)} rows={2} required/>
      <div className="form-actions">
        <label className="timeline-check"><input type="checkbox" checked={timeline}
          onChange={e => setTimeline(e.target.checked)}/>{t('整理事件時間線')}</label>
        {current.status === 'running'
          ? <button type="button" className="quiet" onClick={research.cancel}>{t('停止調查')}</button>
          : <button type="submit" className="primary" disabled={!question.trim()}>{t('開始研究')} <span aria-hidden="true">→</span></button>}
      </div>
    </form>}
  </section>;

  return <div className="shell">
    <header className="topbar"><a className="brand" href="/"><img src="/favicon.svg" alt="" width="28" height="28"/>Trace Atlas</a>
      <div className="header-actions">
        <button type="button" className="quiet history-toggle" aria-label={t('歷史調查')} title={t('歷史調查')} aria-expanded={historyOpen}
          onClick={() => setHistoryOpen(!historyOpen)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7v4h4M4.4 10a8 8 0 1 1 1.1 6M12 7.5v5l3 1.7"/></svg><span>{t('歷史調查')}</span></button>
        <button className="quiet new-run" aria-label={t('新調查')} title={t('新調查')} disabled={current.status === 'running'} onClick={() => {
          research.newConversation(); setQuestion(''); showCurrent();
        }}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg><span>{t('新調查')}</span></button>
        <LanguageMenu locale={locale} onChange={setLocale}/>
      </div>
    </header>
    <div className="workspace">
      <aside className={`history-rail ${historyOpen ? 'is-open' : ''}`} aria-label={t('歷史調查')}>
        <div className="history-title"><h2>{t('歷史調查')}</h2><button className="quiet history-close" type="button"
          onClick={() => setHistoryOpen(false)}>{t('關閉')}</button></div>
        {research.isLoading && <p className="muted">{t('載入中…')}</p>}
        {!research.isLoading && !research.conversations.length && <p className="muted">{t('完成的調查會出現在這裡。')}</p>}
        <ul className="history-list">{research.conversations.map(item => <li key={item.id}>
          <button type="button" className="history-item" aria-current={(preview?.id || research.currentId) === item.id ? 'true' : undefined}
            onClick={() => void openHistory(item.id)}>
            <strong>{item.title === 'New Conversation' ? t('新調查') : item.title}</strong>
            {item.updatedAt && <small>{historyDate.format(item.updatedAt)}</small>}
          </button>
        </li>)}</ul>
        {research.saveError && <p className="muted" role="status">{t('儲存暫時失敗')}</p>}
      </aside>
      <main>
      {composerFirst && composer}
      {historyError && <p className="error" role="alert">{historyError}</p>}
      {!!shown.rounds.length && <nav className="rounds" aria-label={t('對話中的調查輪次')}>
        {[...shown.rounds, shown.current].map((entry, index) => <button key={index} type="button"
          title={entry.question} aria-pressed={(round ?? shown.rounds.length) === index}
          onClick={() => { setRound(index === shown.rounds.length ? undefined : index); setChoice(''); setDetailsOpen(false); setView('map'); }}>
          {entry.question || t('本輪調查')}
        </button>)}
      </nav>}
      <section className="workbench" data-idle={!state.question || undefined} aria-label={t('調查進度')}>
        <div className="workbench-top">
          <div className="research-heading">
            <div className={`activity ${state.status}`} role="status">{currentLabel}</div>
            <h1>{state.question || t('研究路線')}</h1>
          </div>
          <div className="view-buttons">
            <button type="button" aria-pressed={view === 'map'} onClick={() => setView('map')}>{t('線索圖')}</button>
            {(state.timeline || digest) && <button type="button" aria-pressed={view === 'timeline'}
              onClick={() => setView('timeline')}>{t('時間線')}</button>}
            <button type="button" aria-pressed={view === 'report'} disabled={!article} onClick={() => setView('report')}>{t('報告')}</button>
          </div>
        </div>
        {state.error && <div className="error" role="alert">{state.error}</div>}
        {state.timelineError && <div className="error" role="status">{t('時間線生成遇到問題，報告已可查看。')}</div>}
        {!viewingHistory && (state.status === 'failed' || state.status === 'cancelled') && state.question &&
          <div className="recovery">
            <button className="quiet" onClick={() => void research.retry()}>{t('重試調查')}</button>
            {state.recovery?.can_write_report &&
              <button className="quiet" onClick={() => void research.recover('write_report')}>{t('用現有資料寫報告')}</button>}
          </div>}
        {view === 'map' ? <div className={`map-layout${detailsOpen ? ' has-details' : ''}`} onKeyDown={e => {
          if (detailsOpen && e.key === 'Escape') { e.stopPropagation(); closeDetails(); }
        }}>
          <button type="button" id="journey-details-toggle" className="details-toggle" aria-expanded={detailsOpen}
            aria-label={detailsLabel} title={detailsLabel} aria-controls="journey-details" onClick={() => {
              if (detailsOpen) closeDetails();
              else selectNode(selectedNode?.id || journey.active || journey.rows[0][0].id);
            }}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m11 7-5 5 5 5m7-10-5 5 5 5"/></svg>
          </button>
          <JourneyDetails open={detailsOpen} node={selectedNode} state={state} timeline={digest} select={selectNode}
            seeReport={() => setView('report')} seeTimeline={() => setView('timeline')}/>
          <ResearchJourney journey={journey} choice={detailsOpen ? choice : ''} select={selectNode}/>
        </div> : view === 'timeline' ? <Timeline state={{ ...state, timelineView: digest }}
          regroup={runId && state.status === 'completed' && state.timeline?.length ? () => void regroup() : undefined}
          busy={!!digestRequest && digestRequest.runId === runId && !digestRequest.error}
          error={digestRequest && digestRequest.runId === runId ? digestRequest.error : undefined}/> : <section className="report-panel">
          <div className="report-top"><small>{state.article ? t('完整報告') : t('撰寫中的草稿')}</small>
            <button className="quiet" onClick={() => download(article)}>{t('下載 Markdown')}</button></div>
          <article><Report article={article}/></article>
        </section>}
      </section>
      {!composerFirst && composer}
      </main>
    </div>
  </div>;
}

migrateBrowserStorage();
createRoot(document.getElementById('atlas-root')!).render(<Atlas/>);
