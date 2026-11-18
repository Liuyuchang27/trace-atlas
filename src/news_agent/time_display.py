"""Display recorded temporal roles without deriving dates or validity intervals."""

DATE_NOTE = (
    "事件日期表示发生或声明的时间；结论适用时段表示该说法针对的时间范围。"
    "仅在已记录明确依据时展示适用时段；未列出不表示事件日期未知，也不表示一直有效。"
    "网页发布日期单独列在来源信息中，不自动作为事件日期。"
)


def span_label(span):
    precision = span.get("precision", "unknown")
    if precision == "unknown":
        return "日期待确认（当前发现尚未绑定日期依据）"
    start, end = span["start"], span.get("end")
    if end and end != start:
        return start + " 至 " + end
    return start + {"year": "（精确到年）", "month": "（精确到月）"}.get(precision, "")


def time_labels(event_time, valid_time):
    """A missing validity field is unspecified, not proof of inapplicability."""
    return {
        "event": span_label(event_time),
        "applicability": (
            span_label(valid_time) if valid_time.get("precision", "unknown") != "unknown" else None
        ),
        "applicability_note": "未单独记录适用时段；不代表事件日期未知或结论一直有效。",
    }


def finding_time_line(event_time, valid_time):
    labels = time_labels(event_time, valid_time)
    text = "事件日期：" + labels["event"]
    if labels["applicability"]:
        text += "；结论适用时段：" + labels["applicability"]
    return text
