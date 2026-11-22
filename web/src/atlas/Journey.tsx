import { useLayoutEffect, useRef, useState } from 'react';
import { activityDetail, activityLabel, JourneyEdge, JourneyNode, NodeKind, roundedPath } from './journey-model';
import { useLocale } from './locale';

const glyphs: Record<NodeKind, string> = {
  question: 'M5 5h14v11h-7l-4 3v-3H5z M9 9a2 2 0 1 1 3 2v1 M12 14h.01',
  plan: 'M7 5h10 M7 10h10 M7 15h6 M7 20h8 M3 5h.01 M3 10h.01 M3 15h.01 M3 20h.01',
  dispatch: 'M12 5v4m0 0-6 5v5m6-10 6 5v5 M3 19h6 M15 19h6',
  branch: 'M16 16l4 4 M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13z',
  review: 'M6 5v4l6 5m6-9v4l-6 5v5 M3 5h6 M15 5h6',
  writing: 'M6 4h8l4 4v12H6z M14 4v4h4 M9 12h6 M9 16h6',
  time: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 7v5l3 2',
  delivery: 'M5 12l4 4L19 6',
};

export function JourneyGlyph({ kind }: { kind: NodeKind }) {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={glyphs[kind]} fill="none"
    stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function GraphNode({ node, selected, select, incoming, outgoing }: {
  node: JourneyNode; selected: boolean; select: (id: string) => void; incoming: boolean; outgoing: boolean;
}) {
  const t = useLocale();
  const activity = node.status === 'active' && [activityLabel(node.activity, t), activityDetail(node.activity, t)].filter(Boolean).join(' · ');
  return <button className={`graph-node ${node.status} ${node.kind === 'branch' ? 'branch-node' : 'stage-node'}`}
    type="button" id={`journey-node-${node.id}`} data-graph-node={node.id} aria-pressed={selected}
    aria-current={node.status === 'active' ? 'step' : undefined} aria-controls="journey-details" onClick={() => select(node.id)}>
    {incoming && <span className="graph-port port-in" data-port="in" aria-hidden="true"/>}
    <span className="graph-glyph"><JourneyGlyph kind={node.kind}/></span>
    <span className="node-copy">
      <strong>{node.title}</strong>
      <span className="node-preview">{activity || node.preview}</span>
    </span>
    <span className="node-chevron" aria-hidden="true">›</span>
    {outgoing && <span className="graph-port port-out" data-port="out" aria-hidden="true"/>}
  </button>;
}

export function ResearchJourney({ journey, choice, select }: {
  journey: { rows: JourneyNode[][]; edges: JourneyEdge[]; active?: string };
  choice: string; select: (id: string) => void;
}) {
  const t = useLocale();
  const graph = useRef<HTMLDivElement>(null);
  const [geometry, setGeometry] = useState({ width: 1, height: 1,
    paths: [] as { key: string; d: string; status: JourneyEdge['status'] }[] });
  const incoming = new Set(journey.edges.map(edge => edge.to));
  const outgoing = new Set(journey.edges.map(edge => edge.from));
  const edgeSignature = JSON.stringify(journey.edges);

  useLayoutEffect(() => {
    const container = graph.current;
    if (!container) return;
    const nodes = Array.from(container.querySelectorAll<HTMLButtonElement>('[data-graph-node]'));
    let frame = 0;
    function update() {
      if (!container) return;
      const origin = container.getBoundingClientRect();
      const layout = new Map(nodes.map(node => {
        const ports = Object.fromEntries(Array.from(node.querySelectorAll<HTMLElement>('[data-port]')).map(port => {
          const box = port.getBoundingClientRect();
          return [port.dataset.port, { x: box.left + box.width / 2 - origin.left, y: box.top + box.height / 2 - origin.top }];
        }));
        const branchRow = node.parentElement?.classList.contains('graph-branches')
          ? node.parentElement.getBoundingClientRect() : undefined;
        return [node.dataset.graphNode, { ports, branchRow }];
      }));
      const paths = (JSON.parse(edgeSignature) as JourneyEdge[]).flatMap(({ from, to, status }) => {
        const source = layout.get(from), target = layout.get(to);
        const a = source?.ports.out, b = target?.ports.in;
        if (!a || !b) return [];
        const middle = (a.y + b.y) / 2;
        let points = [a, { x: a.x, y: middle }, { x: b.x, y: middle }, b];

        // Wrapped branches route through the side gutters so their wires stay outside other cards.
        const nextRowTop = target?.branchRow && target.branchRow.top - origin.top;
        const prevRowBottom = source?.branchRow && source.branchRow.bottom - origin.top;
        if (nextRowTop !== undefined && b.y > nextRowTop + 3) {
          const rail = b.x < origin.width / 2 ? 9 : origin.width - 9;
          const leave = (a.y + nextRowTop) / 2;
          points = [a, { x: a.x, y: leave }, { x: rail, y: leave }, { x: rail, y: b.y - 8 }, { x: b.x, y: b.y - 8 }, b];
        } else if (prevRowBottom !== undefined && a.y < prevRowBottom - 3) {
          const rail = a.x < origin.width / 2 ? 16 : origin.width - 16;
          const join = (b.y + prevRowBottom) / 2;
          points = [a, { x: a.x, y: a.y + 8 }, { x: rail, y: a.y + 8 }, { x: rail, y: join }, { x: b.x, y: join }, b];
        }
        return [{ key: `${from}-${to}`, d: roundedPath(Math.abs(a.x - b.x) < .25 && points.length === 4 ? [a, b] : points), status }];
      });
      setGeometry({ width: origin.width, height: origin.height, paths });
    }
    function schedule() {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(update);
    }
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(schedule);
    if (observer) {
      observer.observe(container);
      nodes.forEach(node => observer.observe(node));
    } else window.addEventListener('resize', schedule);
    update();
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', schedule);
      window.cancelAnimationFrame(frame);
    };
  }, [edgeSignature]);

  return <section className="journey-panel" aria-label={t('調查線索圖，由上至下')}>
    {journey.active && <div className="journey-toolbar"><button type="button" className="quiet locate-step"
      onClick={() => document.getElementById(`journey-node-${journey.active}`)?.scrollIntoView({ block: 'nearest' })}>{t('定位目前節點')} ↓</button></div>}
    <div className="journey-graph" ref={graph}>
      <svg className="graph-wires" viewBox={`0 0 ${geometry.width} ${geometry.height}`} aria-hidden="true">
        {geometry.paths.map(path => <path key={path.key} className={`graph-edge ${path.status}`} d={path.d}/>)}
      </svg>
      {journey.rows.map(row => <div key={row[0].id} className={`graph-row${row.length > 1 ? ' graph-branches' : ''}`}>
        {row.map(node => <GraphNode key={node.id} node={node} selected={choice === node.id} select={select}
          incoming={incoming.has(node.id)} outgoing={outgoing.has(node.id)}/>)}
      </div>)}
    </div>
  </section>;
}
