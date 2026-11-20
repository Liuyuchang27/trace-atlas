"""Readable projections of saved findings. The full report and audit stay in the run."""

from html import escape
from itertools import groupby
from urllib.parse import urlsplit

from .current_check import checked_sources, related_preorder_notes, result_view
from .current_report import synthesize_notes
from .event_graph import Finding
from .event_report import precision_issue, resolved_temporal_candidate
from .temporal_review import coverage_view, digest


def verified_review(review, graph):
    return (
        review.get("status") == "reviewed"
        and review.get("graph_sha256") == graph.get("sha256")
        and review.get("sha256") == digest({k: v for k, v in review.items() if k != "sha256"})
    )


def current_brief(state, documents, passages, draft=None):
    return {
        "current": result_view(state),
        "sources": checked_sources(state, documents, passages),
        "preorder_notes": synthesize_notes(related_preorder_notes(state), draft),
        "previous_period": state.get("previous_period", {}),
        "searches": [
            {
                "from": row["from"],
                "latest_update_from": row.get("latest_update_from"),
                "domains": row["domains"],
            }
            for row in state["attempts"]
        ],
    }


def report_brief(snapshot, review, *, repeat_review_incomplete=False):
    graph = snapshot["event_graph"]
    source_passages = {p["passage_id"]: (s, p) for s in snapshot["sources"] for p in s["passages"]}
    units = snapshot["event_view"]["event_units"]
    decisions = {d["finding_id"]: d for d in review.get("decisions", [])}
    plan = {
        "temporal_review": review,
        "temporal_repairs": list(snapshot["temporal_repairs"].get("targets", {}).values()),
        "graph_choices": [
            {
                "finding_id": u["event_unit_id"],
                **u["finding"],
                "passage_ids": [r["passage_id"] for r in u["references"]],
            }
            for u in units
        ],
    }
    findings = []
    for unit in units:
        refs = unit["references"]
        if not refs or any(
            r["passage_id"] not in source_passages
            or source_passages[r["passage_id"]][0]["version_id"] != r["version_id"]
            or source_passages[r["passage_id"]][1]["text"][
                r["start"] - source_passages[r["passage_id"]][1]["start"] : r["end"]
                - source_passages[r["passage_id"]][1]["start"]
            ]
            != r["quote"]
            for r in refs
        ):
            continue
        finding = unit["finding"]
        if precision_issue(Finding.model_validate(finding)):
            continue
        fid = unit["event_unit_id"]
        node_id = unit["question_node_id"]
        row = {
            "id": fid,
            "node_id": node_id,
            "part_id": finding.get("question_part_id"),
            "text": finding["assertion"],
            "date": finding["event_time"].get("start"),
            "end": finding["event_time"].get("end"),
            "valid_time": unit["time_labels"]["applicability"],
            "occurrence": finding["occurrence"],
            "stance": finding["stance"],
            "declared_answer": finding["answers_question_part"],
            "passages": list(dict.fromkeys(r["passage_id"] for r in refs)),
            "verdict": decisions.get(fid, {}).get("event_verdict"),
        }
        if resolved := resolved_temporal_candidate(plan, fid, "event_date_basis"):
            candidate, _ = resolved
            if set(candidate["passage_ids"]) <= source_passages.keys():
                row.update(
                    date=candidate["event_time"].get("start"),
                    end=candidate["event_time"].get("end"),
                    passages=list(dict.fromkeys([*row["passages"], *candidate["passage_ids"]])),
                    verdict="supported",
                )
        findings.append(row)
    if not findings:
        return None

    coverage = {(d["node_id"], d["question_part_id"]): d for d in coverage_view(review, graph)}
    answers, charts, omitted, used = [], [], [], set()
    visible = {row["id"]: row for row in findings}
    searched_months = {
        (r["node_id"], r.get("question_part_id"), r["reference_month"])
        for r in snapshot.get("adaptive", {}).get("rounds", [])
        if r.get("reference_month") and r.get("status") == "closed"
    }
    for node in snapshot["event_view"]["nodes"]:
        for part in node.get("part_status", []):
            part_id = part["part_id"]
            if plan := part.get("month_coverage"):
                points = [
                    {
                        "month": month,
                        "searched": (node["node_id"], part_id, month) in searched_months,
                        "value": point["value"] if point.get("finding_id") in visible else None,
                        "passages": visible[point["finding_id"]]["passages"]
                        if point.get("finding_id") in visible
                        else [],
                    }
                    for month in plan["months"]
                    for point in [plan["points"].get(month, {})]
                ]
                if points:
                    receipt = next(
                        (
                            row
                            for row in snapshot.get("adaptive", {}).get("series_attempts", [])
                            if row["node_id"] == node["node_id"]
                            and row["question_part_id"] == part_id
                            and row["status"] == "completed"
                            and {p["finding_id"] for p in row["points"]}
                            == {p["finding_id"] for p in plan["points"].values()}
                            and all(p["value"] is not None for p in points)
                        ),
                        None,
                    )
                    charts.append(
                        {
                            "question": part["question"],
                            "unit": part["monthly_series"]["unit"],
                            "points": points,
                            **(
                                {"source": receipt["url"], "publisher": receipt["publisher"]}
                                if receipt
                                else {}
                            ),
                        }
                    )
                    used.update(p["finding_id"] for p in plan["points"].values())
                else:
                    omitted.append(part["question"])
                continue
            decision = coverage.get((node["node_id"], part_id), {})
            checked = set(decision.get("finding_ids", []))
            choices = [
                r
                for r in findings
                if r["node_id"] == node["node_id"]
                and r["part_id"] == part_id
                and (r["declared_answer"] or r["id"] in checked)
                and r["stance"] in {"supports", "contradicts"}
                and part["required_occurrence"] in {"any", r["occurrence"]}
                and (
                    not part["requires_date"]
                    or (r["date"] and r["verdict"] not in {"needs_evidence", "conflicting"})
                )
            ]
            choices.sort(
                key=lambda r: (r["id"] in checked, r["verdict"] == "supported", bool(r["date"])),
                reverse=True,
            )
            if not choices or decision.get("verdict") == "unsupported":
                omitted.append(part["question"])
                continue
            picked = [choices[0]]
            if part["status"] == "disputed" or decision.get("verdict") == "conflicting":
                counter = next((r for r in choices if r["stance"] != picked[0]["stance"]), None)
                if counter:
                    picked.append(counter)
            label = (
                "来源有分歧"
                if len(picked) > 1 or decision.get("verdict") == "conflicting"
                else "部分回答"
                if decision.get("verdict") == "partial"
                else None
            )
            answers.append({"question": part["question"], "rows": picked, "status": label})
            used.update(r["id"] for r in picked)

    dated, background = [], {}
    for row in findings:
        if row["id"] in used:
            continue
        if row["date"] and row["verdict"] not in {"needs_evidence", "conflicting"}:
            # A separate finding for the same part/date still appears under supporting records.
            if not any(
                r["id"] in used
                and r["node_id"] == row["node_id"]
                and r["part_id"] == row["part_id"]
                and r["date"] == row["date"]
                and r["stance"] == row["stance"]
                for r in findings
            ):
                dated.append(row)
                continue
        key = (row["node_id"], row["part_id"], row["date"], row["stance"] == "contradicts")
        group = background.setdefault(key, [])
        if not any(r["text"] == row["text"] and r["passages"] == row["passages"] for r in group):
            group.append(row)

    timeline_rows = [
        r
        for a in answers
        for r in a["rows"]
        if r["date"] and r["verdict"] not in {"needs_evidence", "conflicting"}
    ] + dated
    timeline = [
        {"date": date, "ids": [r["id"] for r in group]}
        for date, group in groupby(
            sorted(timeline_rows, key=lambda r: r["date"]), key=lambda r: r["date"]
        )
    ]
    sources = {}
    for row in findings:
        for pid in row["passages"]:
            source, _ = source_passages[pid]
            item = sources.setdefault(
                source["version_id"], {"title": source["title"], "url": source["url"], "cited": []}
            )
            if pid not in item["cited"]:
                item["cited"].append(pid)
    return {
        "answers": answers,
        "charts": charts,
        "dated": dated,
        "timeline": timeline,
        "background": list(background.values()),
        "sources": list(sources.values()),
        "omitted": omitted,
        "repeat_review_incomplete": repeat_review_incomplete,
        "has_review": bool(decisions),
    }


def render_brief(brief, *, standalone=False):
    def cite(pid, label="查看原文"):
        pid = escape(pid, quote=True)
        attr = "" if standalone else f' data-passage="{pid}"'
        return f'<a class="brief-citation" href="#{pid}"{attr}>{label}</a>'

    def finding(row):
        clock = ""
        if row["date"]:
            value = row["date"] + (f" 至 {row['end']}" if row["end"] else "")
            verdict = " · 日期有疑问" if row["verdict"] in {"needs_evidence", "conflicting"} else ""
            clock = f"<time>{escape(value + verdict)}</time>"
        kind = {"planned": "计划", "statement": "来源陈述"}.get(row["occurrence"])
        if kind:
            clock += f'<span class="brief-kind">{kind}</span>'
        validity = (
            f"<small>适用时段 {escape(row['valid_time'])}</small>" if row["valid_time"] else ""
        )
        links = " ".join(cite(p, f"原文 {i}") for i, p in enumerate(row["passages"], 1))
        return (
            f'<div id="{escape(row["id"], quote=True)}" class="brief-finding">{clock}'
            f"<p>{escape(row['text'])}</p>{validity}"
            f'<div class="brief-citations">{links}</div></div>'
        )

    if current := brief.get("current"):
        return render_current(brief, current, cite)

    output = ['<div class="brief">']
    if brief["omitted"] and not brief["answers"] and not brief["charts"]:
        output.append("<section><h2>核心结论未确认</h2><ul>")
        output.extend(f"<li>{escape(question)}</li>" for question in brief["omitted"])
        output.append("</ul></section>")
    if brief["answers"]:
        output.append('<section><h2>核心回答</h2><div class="brief-grid">')
        for answer in brief["answers"]:
            label = f"<small>{answer['status']}</small>" if answer["status"] else ""
            output.append(
                f"<article><h3>{escape(answer['question'])}</h3>{label}"
                + "".join(map(finding, answer["rows"]))
                + "</article>"
            )
        output.append("</div></section>")
    for chart in brief["charts"]:
        plotted = [p["value"] for p in chart["points"] if p["value"] is not None]
        scale = max([1, *(abs(v) for v in plotted)])
        negative = any(v < 0 for v in plotted)
        method = ""
        if chart.get("source"):
            source = escape(chart["source"], quote=True)
            publisher = escape(chart["publisher"])
            unit = escape(chart["unit"])
            increases = sum(value > 0 for value in plotted)
            decreases = sum(value < 0 for value in plotted)
            flat = len(plotted) - increases - decreases
            counts = "、".join(
                f"{count} 个月{direction}"
                for count, direction in ((increases, "增加"), (decreases, "减少"), (flat, "持平"))
                if count
            )
            method = (
                f"<p>{counts}，区间累计净变化 {sum(plotted):+,} {unit}。</p>"
                '<p class="muted">同一版本相邻月水平相减。本次快照可包含后续修订，'
                "各月初值和逐次修订需另查历史版本。 "
                f'<a href="{source}" target="_blank" rel="noopener noreferrer">'
                f"完整原表 · {publisher}</a></p>"
            )
        output.append(
            f"<section><h2>{escape(chart['question'])}</h2>"
            f'<p class="muted">单位 {escape(chart["unit"])}'
            f' · 按数据所属月排列</p>{method}<div class="brief-series">'
        )
        for point in chart["points"]:
            value = point["value"]
            title = (
                ("已搜待补证" if point["searched"] else "未按月检索")
                if value is None
                else format(value, "+,d")
                if value
                else "0"
            )
            value_class = ' class="series-missing"' if value is None else ""
            width = 0 if value is None else abs(value) * (50 if negative else 100) / scale
            start = 50 - width if value is not None and value < 0 else 50 if negative else 0
            bar = (
                f'<i class="series-bar{" down" if value < 0 else ""}" '
                f'style="left:{start:.3f}%;width:{width:.3f}%"></i>'
                if value is not None
                else ""
            )
            labels = ("上月原值", "本月原值", "口径") if method else ()
            links = " ".join(
                cite(pid, labels[i - 1] if i <= len(labels) else f"原文 {i}")
                for i, pid in enumerate(point["passages"], 1)
            )
            output.append(
                f'<div class="series-row"><time>{escape(point["month"])}</time>'
                f'<span class="series-track" aria-hidden="true">{bar}</span>'
                f"<strong{value_class}>{title}</strong>"
                f'<span class="series-sources">{links}</span></div>'
            )
        output.append("</div></section>")
    if brief["timeline"]:
        output.append('<section><h2>时间索引</h2><ol class="brief-timeline">')
        for group in brief["timeline"]:
            anchors = " ".join(
                f'<a href="#{escape(fid, quote=True)}" '
                f'aria-label="查看此日期的第 {i} 条记录">{i}</a>'
                for i, fid in enumerate(group["ids"], 1)
            )
            output.append(f"<li><time>{escape(group['date'])}</time><div>{anchors}</div></li>")
        output.append("</ol></section>")
    if brief["dated"]:
        output.append('<section><h2>事件与背景</h2><div class="brief-grid">')
        output.extend("<article>" + finding(row) + "</article>" for row in brief["dated"])
        output.append("</div></section>")
    if brief["repeat_review_incomplete"]:
        output.append(
            '<p class="brief-note">补充日期复审未完成。已有日期审核沿用自同一份证据记录。'
            "缺少年份的材料见补充记录。</p>"
        )
    elif not brief["has_review"]:
        output.append(
            '<p class="brief-note">日期采用已保存的原文记录，独立步骤中的模型复审尚未完成。</p>'
        )
    if brief["omitted"] and (brief["answers"] or brief["charts"]):
        output.append("<section><h2>仍待回答</h2><ul>")
        output.extend(f"<li>{escape(question)}</li>" for question in brief["omitted"])
        output.append("</ul></section>")
    if brief["background"]:
        output.append("<details><summary>查看补充记录</summary>")
        for group in brief["background"]:
            title = "日期未明确" if not group[0]["date"] else "补充证据"
            output.append(f"<details><summary>{title} · {len(group)} 条</summary>")
            output.extend(map(finding, group))
            output.append("</details>")
        output.append("</details>")
    output.append(f"<details><summary>引用来源 · {len(brief['sources'])} 个页面</summary><ul>")
    for source in brief["sources"]:
        url = source["url"]
        host = urlsplit(url).hostname if url else ""
        link = (
            f' <a href="{escape(url, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">原站 ↗</a>'
            if url
            else ""
        )
        refs = " ".join(cite(p, f"原文 {i}") for i, p in enumerate(source["cited"], 1))
        output.append(
            f"<li><strong>{escape(source['title'])}</strong> <small>{escape(host or '')}</small>"
            f'{link}<div class="brief-citations">{refs}</div></li>'
        )
    output.append("</ul></details></div>")
    return "".join(output)


def render_current(brief, current, cite):
    confirmed = current["outcome"] == "confirmed"
    heading = current["label"] if confirmed else "核查结论"
    conclusion = [f"<section><h2>{escape(heading)}</h2>"]
    if not confirmed:
        conclusion.append(f'<p class="muted">核查截至 {escape(current["cutoff_label"])}</p>')
    conclusion.append(f"<p>{escape(current['text'])}</p>")
    if current["record_on"]:
        conclusion.append(f"<p>原始记录日期 {escape(current['record_on'])}</p>")
    if cutoff := current.get("data_through"):
        conclusion.append(f"<p>数据截至 {escape(cutoff)}</p>")
    if confirmed and current.get("release_pending"):
        conclusion.append("<p>本轮未取得更新的发售状态公告。</p>")
    if confirmed:
        conclusion.append(f'<p class="muted">核查截至 {escape(current["cutoff_label"])}</p>')
    conclusion.append("</section>")
    findings = []
    if notes := brief.get("preorder_notes"):
        findings.append("<section><h2>已查到的预购信息</h2>")
        for note in notes:
            text = escape(note.get("summary", note["text"]))
            findings.append(
                f"<p><strong>{escape(note['label'])}</strong></p><p>{text} "
                f"{cite(note['passage_id'])}</p>"
            )
        findings.append("</section>")
    if release := current.get("release"):
        refs = " ".join(cite(r["passage_id"], "发售公告原文") for r in release["references"])
        findings.append(
            f"<section><p>官方 {escape(release['day'])} 公告确认已发售。 {refs}</p></section>"
        )
    output = ['<div class="brief">']
    output.extend(conclusion + findings if confirmed else findings + conclusion)
    if brief["sources"]:
        output.append("<section><h2>已读取的来源</h2><ul>")
        for source in brief["sources"]:
            refs = " ".join(cite(pid, f"原文 {i}") for i, pid in enumerate(source["cited"], 1))
            output.append(
                f'<li><a href="{escape(source["url"], quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">{escape(source["title"])}</a> {refs}</li>'
            )
        output.append("</ul></section>")
    if brief["searches"]:
        output.append("<details><summary>核查范围</summary><ul>")
        for row in brief["searches"]:
            domains = "、".join(map(escape, row["domains"]))
            from_day = escape(row["from"])
            end_day = escape(current["cutoff"])
            update = row.get("latest_update_from")
            scope = (
                f"进展 {escape(update)} 至 {end_day}，所问数据 {from_day} 至 {end_day}"
                if update
                else f"{from_day} 至 {end_day}"
            )
            output.append(f"<li>{scope} · {domains}</li>")
        if (previous := brief.get("previous_period", {})).get("from"):
            output.append(
                f"<li>上一披露时段 {escape(previous['from'])} 至 {escape(current['cutoff'])}</li>"
            )
        output.append("</ul></details>")
    return "".join(output) + "</div>"
