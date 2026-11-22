import type { ResearchActivity, ResearchState, ResearchTask } from '@/research/state';

import type { MessageKey, Translate } from './i18n';

const stageLabels: Record<string, MessageKey> = {
  connecting: '連接研究引擎', initializing: '理解研究範圍', planning: '規劃調查方向',
  searching: '檢索資料', filtering: '篩選來源', reading: '閱讀原文', updating_graph: '核對並匯合證據',
  planning_outline: '整理報告大綱', planning_sections: '整理各節證據', writing_section: '撰寫報告',
  writing_summary: '撰寫摘要', finalizing: '核對引用', building_timeline: '整理事件日期', grouping_timeline: '歸納關鍵時段',
};
const writing = new Set(['planning_outline', 'planning_sections', 'writing_section', 'writing_summary', 'finalizing']);
const timeline = new Set(['building_timeline', 'grouping_timeline']);
const taskLabels = { pending: '等待取證', running: '取證中', completed: '已交回', failed: '取證失敗', cancelled: '已停止' } as const;

export type NodeStatus = 'done' | 'active' | 'pending' | 'stopped';
export type NodeKind = 'question' | 'plan' | 'dispatch' | 'branch' | 'review' | 'writing' | 'time' | 'delivery';
export type JourneyNode = {
  id: string;
  kind: NodeKind;
  title: string;
  status: NodeStatus;
  preview?: string;
  activity?: ResearchActivity;
  tasks?: ResearchTask[];
  task?: ResearchTask;
};
export type JourneyEdge = { from: string; to: string; status: NodeStatus };

export function activityLabel(activity: ResearchActivity | undefined, t: Translate) {
  const key = stageLabels[activity?.stage || ''];
  return key ? t(key) : '';
}

export function activityDetail(activity: ResearchActivity | undefined, t: Translate) {
  if (!activity) return '';
  if (activity.query) return activity.query;
  if (activity.section_total && activity.section_index) {
    return t('第 {i}／{n} 節', { i: activity.section_index, n: activity.section_total }) + (activity.title ? ' · ' + activity.title : '');
  }
  return activity.title || (activity.events ? t('{n} 條事件記錄', { n: activity.events }) : '');
}

export function branchTitle(task: ResearchTask) {
  return task.question.split(/[：:。！？!?]/, 1)[0].trim() || task.question;
}

function batches(state: ResearchState) {
  const known = new Set<string>();
  const groups = (state.taskBatches || []).map(ids => ids.flatMap(id => {
    const task = state.tasks[id];
    if (!task || known.has(id)) return [];
    known.add(id);
    return [task];
  })).filter(batch => batch.length);
  const remaining = Object.values(state.tasks).filter(task => !known.has(task.id));
  return remaining.length ? [...groups, remaining] : groups;
}

export function buildJourney(state: ResearchState, t: Translate) {
  const stage = state.activity?.stage || '';
  const active = state.status === 'running';
  const end = state.status === 'completed';
  const groups = batches(state);
  const editing = writing.has(stage);
  const timing = timeline.has(stage);
  const reportStarted = !!(state.article || state.partialArticle) || (active && (editing || timing));
  const timelineStarted = (active && timing) || !!state.timelineView || !!state.timeline?.length;
  const reviewing = active && !reportStarted && (stage === 'updating_graph' || stage === 'planning');
  const edges: JourneyEdge[] = [];
  const rows: JourneyNode[][] = [[{ id: 'question', kind: 'question', title: t('研究問題'),
    status: state.question ? 'done' : 'pending', preview: state.question || t('等待輸入問題') }]];
  let previous = 'question';

  function step(node: JourneyNode) {
    edges.push({ from: previous, to: node.id, status: node.status });
    rows.push([node]);
    previous = node.id;
  }

  if (state.question) {
    const status = groups.length || reportStarted ? 'done' : active ? 'active' : 'stopped';
    step({ id: 'plan', kind: 'plan', title: t('規劃調查'), status, tasks: groups[0],
      ...(status === 'active' ? { activity: state.activity } : {}),
      preview: groups.length ? t('首輪安排 {n} 條查證方向', { n: groups[0].length }) : undefined });
  }

  groups.forEach((tasks, index) => {
    const branches: JourneyNode[] = tasks.map(task => {
      const activity = state.taskActivity?.[task.id] || task.activity;
      const status = task.status === 'completed' ? 'done' : ['cancelled', 'failed'].includes(task.status) ? 'stopped'
        : active && (task.status === 'running' || activity) ? 'active' : 'pending';
      return { id: task.id, kind: 'branch', title: branchTitle(task), status, task, activity,
        preview: status === 'active' ? activityLabel(activity, t) || t(taskLabels.running) : t(taskLabels[task.status]) };
    });
    const settled = tasks.filter(task => !['pending', 'running'].includes(task.status)).length;
    const dispatch = `dispatch-${index}`;
    step({ id: dispatch, kind: 'dispatch', tasks,
      title: index ? t('追查新線索') : tasks.length > 1 ? t('並行查證') : t('查證與閱讀'),
      status: active && !settled && !branches.some(node => node.status === 'active') ? 'active' : 'done',
      preview: t('{done}／{total} 條線索已收尾', { done: settled, total: tasks.length }) });
    rows.push(branches);
    for (const branch of branches) edges.push({ from: dispatch, to: branch.id, status: branch.status });

    if (settled === tasks.length || index < groups.length - 1 || reportStarted) {
      const review = `review-${index}`;
      const status = index === groups.length - 1 && reviewing ? 'active'
        : branches.some(node => node.status === 'done') ? 'done' : 'stopped';
      rows.push([{ id: review, kind: 'review', tasks, status,
        title: index < groups.length - 1 ? t('整理本輪發現') : t('匯合證據'),
        preview: status === 'active' ? t('核對發現，決定是否補查') : undefined }]);
      for (const branch of branches) edges.push({ from: branch.id, to: review,
        status: status === 'active' ? 'active' : branch.status });
      previous = review;
    }
  });

  if (reportStarted) step({ id: 'writing', kind: 'writing', title: t('撰寫報告'),
    status: active && editing ? 'active' : state.article || timing ? 'done' : 'stopped',
    activity: editing ? state.activity : undefined });
  if (timelineStarted) step({ id: 'time', kind: 'time', title: t('整理時間線'),
    status: active && timing ? 'active' : 'done', activity: timing ? state.activity : undefined,
    preview: state.timelineView?.groups.length ? t('{n} 個研究時段', { n: state.timelineView.groups.length }) : undefined });
  if (end && state.article) step({ id: 'delivery', kind: 'delivery', title: t('報告完成'), status: 'done' });
  return { rows, edges, active: rows.flat().find(node => node.status === 'active')?.id };
}

type Point = { x: number; y: number };

export function roundedPath(points: Point[]) {
  const vertices = points.filter(({x, y}, i) => !i || x !== points[i - 1].x || y !== points[i - 1].y);
  const [first, ...rest] = vertices;
  const segments = [`M ${first.x} ${first.y}`];
  rest.forEach((point, index) => {
    const before = vertices[index];
    const after = rest[index + 1];
    if (!after) {
      segments.push(`L ${point.x} ${point.y}`);
      return;
    }
    const radius = Math.min(8, Math.hypot(before.x - point.x, before.y - point.y) / 2,
      Math.hypot(after.x - point.x, after.y - point.y) / 2);
    const approach = { x: point.x + Math.sign(before.x - point.x) * radius, y: point.y + Math.sign(before.y - point.y) * radius };
    const exit = { x: point.x + Math.sign(after.x - point.x) * radius, y: point.y + Math.sign(after.y - point.y) * radius };
    segments.push(`L ${approach.x} ${approach.y} Q ${point.x} ${point.y} ${exit.x} ${exit.y}`);
  });
  return segments.join(' ');
}
