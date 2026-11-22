export type TaskStatus = 'pending' | 'running' | 'failed' | 'completed' | 'cancelled';
export type ResearchActivity = {stage: string; title?: string; query?: string; section_index?: number; section_total?: number; events?: number};
export type Source = {url: string; title?: string; snippet?: string; evidence_quote?: string};
type Sources = Record<string,Source>;
export type ResearchTask = {id: string; question: string; status: TaskStatus; activity?: ResearchActivity;
  steps?: Array<({action:'search'; query:string} | {action:'read'; focus:string}) & {reason?:string; count:number}>;
  handoff?: {kind:'agent'|'loop_guard'; reason:string; gap:string};
  result?: {answer: string; references: Sources; evidence_records?: Source[]; error?: string} | null};
export type TimelineObservation = {date:string; date_end?:string; role:string; event:string; date_basis?:string;
  evidence:{source_url:string; quote:string; source_date?:string}[]};
export type TimelineView = {groups:{title:string; start:string; end:string; events:TimelineObservation[]; summary?:string}[];
  undated:TimelineObservation[]};
type Recovery = {run_id:string; status:string; can_continue:boolean; can_write_report:boolean; language?:string};
export type ResearchState = {question:string; status:'idle'|'running'|'completed'|'failed'|'cancelled'; tasks:Record<string,ResearchTask>;
  taskBatches?:string[][]; activity?:ResearchActivity; taskActivity?:Record<string,ResearchActivity>; article:string; partialArticle?:string;
  timeline?:TimelineObservation[]; timelineView?:TimelineView; timelineError?:boolean; recovery?:Recovery; references:Sources; error?:string};
export type ResearchSession = {schema_version:2; current:ResearchState; rounds:ResearchState[]};
export function blank(question = ''):ResearchState {
  return {question, status: question ? 'running' : 'idle', article:'', references:{}, tasks:{}};
}
export function sessionOf(current = blank(), rounds:ResearchState[] = []):ResearchSession {
  return {schema_version:2,current,rounds};
}

// The persisted HTTP view and the live event payload share these field names.
function combine(state:ResearchState, data:any):ResearchState {
  const next = {...state};
  if (data.tasks || data.task) {
    next.tasks = {...state.tasks, ...data.tasks};
    if (data.task) next.tasks[data.task.id] = data.task;
  }
  if (data.ref2url) next.references = data.ref2url;
  if (data.task_batches) next.taskBatches = data.task_batches;
  if (data.task_activity) next.taskActivity = data.task_activity;
  if (data.activity) next.activity = data.activity;
  if (data.partial_article) next.partialArticle = data.partial_article;
  if (data.article) next.article = data.article;
  if (data.timeline) next.timeline = data.timeline;
  if (data.timeline_view) next.timelineView = data.timeline_view;
  if ('timeline_error' in data) next.timelineError = data.timeline_error;
  return next;
}

export function restoreSavedResearch(state:ResearchState, saved:any):ResearchState {
  const next = combine(state,saved.view);
  const status = saved.recovery.status;
  return {...next, status:status === 'stopping' ? 'running' : status, error:saved.error || undefined,
    recovery:{...saved.recovery,language:saved.language}};
}
export function advance(state:ResearchState, frame:any):ResearchState {
  if (frame.schema_version !== 2 || !frame.data) throw new Error('Unsupported research stream');
  const {data,type,task_id} = frame;
  const next = combine(state,data);
  switch (type) {
    case 'research.started': return {...next, status:'running', question:data.question};
    case 'research.progress':
      if (task_id) next.taskActivity = {...next.taskActivity,[task_id]:data};
      return {...next,activity:data};
    case 'research.run': case 'research.recovery': return {...next,recovery:{...state.recovery,...data}};
    case 'research.failed': return {...next,status:'failed',error:data.error};
    case 'research.cancelled': return {...next,status:'cancelled'};
    case 'research.completed': return {...next,status:'completed',error:undefined};
    default: return next;
  }
}
