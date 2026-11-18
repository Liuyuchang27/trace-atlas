"""Current source access from saved body/dispatch records, separate from historical prose."""

import re

from .storage import read_json

ACCESS_NOTE = re.compile(
    r"(?:未|尚未)[^。；\n]{0,20}(?:取得|获取|读取|读到)|"
    r"(?:not|never|not yet)\s+(?:been\s+)?(?:read|retrieved|obtained|accessed)",
    re.I,
)


def historical_note(text):
    if text and ACCESS_NOTE.search(text):
        return "历史访问备注（不代表当前读取状态，见来源读取记录）：" + text
    return text


def inventory(evidence):
    adaptive = getattr(evidence, "adaptive", None)
    budget = getattr(adaptive, "budget", None)
    ledger = budget.state if budget else getattr(evidence, "source_access_ledger", None)
    store = getattr(evidence, "store", None)
    if ledger is None and store and (store.root / "budget.json").exists():
        ledger = read_json(store.root / "budget.json")
    delivered = set((ledger or {}).get("delivered_passages", []))
    return [
        {
            "version_id": vid,
            "url": doc.url,
            "title": doc.title,
            "body_saved": bool(doc.content),
            "delivery_known": ledger is not None,
            "delivered_passage_ids": sorted(
                pid
                for pid in delivered
                if pid in evidence.passages and evidence.passages[pid].version_id == vid
            ),
        }
        for vid, doc in evidence.documents.items()
    ]


def render_inventory(evidence):
    lines = [
        "",
        "## 来源读取记录",
        "",
        "以下依据保存正文与工具交付账本更新；交付若干段落不等于通读全文，"
        "读取成功不等于一手来源或事实已获证实。历史访问备注不能覆盖这些记录。",
        "",
    ]
    for row in inventory(evidence):
        state = (
            f"已向 Agent 交付 {len(row['delivered_passage_ids'])} 段原文"
            if row["delivered_passage_ids"]
            else "正文已保存，尚无原文段落交付记录"
            if row["delivery_known"]
            else "正文已保存，当前视图未载入交付账本"
        )
        title = row["title"].replace("]", "\\]").replace("\n", " ")
        lines.append(f"- [{title}]({row['url']})：{state}；版本 {row['version_id']}。")
    return lines
