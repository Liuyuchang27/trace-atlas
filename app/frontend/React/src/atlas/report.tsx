import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Source } from '@/research/state';
import { useLocale } from './locale';

export function webUrl(input?: string) {
  if (!input) return '';
  try {
    const url = new URL(input);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : '';
  } catch { return ''; }
}

export function linkedMarkdown(article: string, sources: Record<string, Source>) {
  return article.replace(/^\[\[(\d+)\]\] (?=\[)/gm, '$1. ')
    .replace(/\[\[(\d+)\]\]/g, (_, id) => {
      const url = webUrl(sources[id]?.url);
      return url ? `[${id}](<${url}>)` : `[${id}]`;
    });
}

function Chart({ markdown }: { markdown: string }) {
  const t = useLocale();
  try {
    const { type, title, unit, labels, values } = JSON.parse(markdown);
    if (!['line', 'bar'].includes(type) || typeof title !== 'string' || typeof unit !== 'string'
        || !Array.isArray(labels) || !Array.isArray(values) || labels.length !== values.length
        || !values.length || labels.some(label => typeof label !== 'string')
        || values.some(value => typeof value !== 'number' || !Number.isFinite(value))) {
      return <code>{markdown}</code>;
    }

    const low = Math.min(0, ...values);
    const high = Math.max(0, ...values);
    const y = (value: number) => 155 - (value - low) / (high - low || 1) * 135;
    const width = Math.max(440, labels.length * 70);
    const step = (width - 48) / labels.length;
    const points = values.map((value, i) => ({ x: 24 + step * (i + .5), y: y(value) }));
    return <figure className="data-chart">
      <figcaption>{title}{unit && <small> · {unit}</small>}</figcaption>
      <div className="chart-scroll"><svg viewBox={`0 0 ${width} 190`} style={{ minWidth: labels.length > 5 ? width : undefined }}
        role="img" aria-label={title}>
        <line x1="18" y1={y(0)} x2={width - 18} y2={y(0)} className="chart-axis"/>
        {type === 'line' && <polyline points={points.map(p => `${p.x},${p.y}`).join(' ')} className="chart-line"/>}
        {points.map((point, i) => <g key={i}>
          <title>{labels[i]} · {values[i]} {unit}</title>
          {type === 'line' ? <circle cx={point.x} cy={point.y} r="4" className="chart-point"/>
            : <rect x={point.x - step * .25} y={Math.min(point.y, y(0))} width={step * .5}
              height={Math.abs(point.y - y(0))} rx="3" className="chart-point"/>}
          <text x={point.x} y="177" textAnchor="middle">{labels[i]}</text>
        </g>)}
      </svg></div>
      <details><summary>{t('查看圖表數據')}</summary><table>
        <thead><tr><th>{t('項目')}</th><th>{unit || t('數值')}</th></tr></thead>
        <tbody>{labels.map((label, i) => <tr key={i}><td>{label}</td><td>{values[i]}</td></tr>)}</tbody>
      </table></details>
    </figure>;
  } catch { return <code>{markdown}</code>; }
}

export function Report({ article }: { article: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml components={{
    a: ({ href, children }) => <a href={webUrl(href) || undefined} target="_blank" rel="noreferrer">{children}</a>,
    img: ({ src, alt }) => webUrl(src) ? <img src={webUrl(src)} alt={alt || ''} loading="lazy" referrerPolicy="no-referrer"/> : null,
    pre: ({ children, node }) => node?.children.some(
      child => child.type === 'element' && Array.isArray(child.properties.className) && child.properties.className.includes('language-chart'),
    ) ? <>{children}</> : <pre>{children}</pre>,
    code: ({ className, children }) => className === 'language-chart'
      ? <Chart markdown={String(children)}/> : <code className={className}>{children}</code>,
  }}>{article}</ReactMarkdown>;
}
