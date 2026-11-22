"""Ground findings in passages and produce the workbench's dated evidence groups."""
import calendar
import re
from datetime import date, datetime, timezone
from urllib.parse import urlsplit, urlunsplit


def url_key(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def date_string(value):
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(value / 1000 if value > 10**11 else value, timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    if not isinstance(value, str):
        return ""
    match = re.fullmatch(r"(\d{4}(?:-\d\d(?:-\d\d)?)?)(?:[T ]\d\d:\d\d.*)?", value.strip())
    if not match:
        return ""
    stamp = match[1]
    try:
        date.fromisoformat((stamp + "-01-01")[:10])
    except ValueError:
        return ""
    return stamp


def passages(text, size=24000):
    # Split transport payloads at paragraph boundaries. Every character reaches a reader.
    while len(text) > size:
        split = text.rfind("\n", size // 2, size)
        split = split + 1 if split >= 0 else size
        yield text[:split]
        text = text[split:]
    if text.strip():
        yield text


def attested_date(stamp, quote):
    if not stamp or stamp[:4] not in quote:
        return ""
    if len(stamp) == 4 or stamp in quote:
        return stamp
    year, month = stamp[:4], int(stamp[5:7])
    month_name = rf"(?:{calendar.month_name[month]}|{calendar.month_abbr[month]}\.?)"
    if len(stamp) == 10:
        day = int(stamp[8:10])
        forms = [rf"{year}\s*年\s*0?{month}\s*月\s*0?{day}\s*[日号號]",
                 rf"{month_name}\s+0?{day}(?:st|nd|rd|th)?[,]?\s+{year}",
                 rf"\b0?{day}\s+{month_name}[,]?\s+{year}"]
        if any(re.search(form, quote, re.I) for form in forms):
            return stamp
    if re.search(rf"{year}\s*年\s*0?{month}\s*月|{month_name}.{{0,12}}\b{year}\b", quote, re.I):
        return stamp[:7]
    return year


def verified(notes, page, hit):
    available = " ".join(page.split())
    seen = set()
    result = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        quote = " ".join(str(note.get("quote") or "").split())
        claim = str(note.get("claim") or "").strip()
        if not claim or len(quote) < 5 or quote not in available or (claim, quote) in seen:
            continue
        seen.add((claim, quote))
        role = note.get("role")
        role = role if role in {"observed", "planned", "announced", "published", "uncertain"} else "uncertain"
        stamp = attested_date(date_string(note.get("date")), quote)
        publication = date_string(hit.get("source_date"))
        basis = "event"
        if not stamp:
            # A publisher timestamp can bound a past event or date the record of a plan.
            stamp = publication
            basis = "documented_by" if role == "observed" else "source_publication"
        elif role == "published":
            basis = "source_publication"
        elif note.get("approximate") is True:
            basis = "approximate"
        if role == "observed" and stamp and stamp > date.today().isoformat():
            role = "uncertain"
        result.append({"claim": claim, "quote": quote, "date": stamp or "unknown", "role": role,
                       "date_basis": basis, "source_date": publication})
    return result


def observations(sources):
    merged = {}
    for source in sources.values():
        for note in source["notes"]:
            key = (note["date"], note["role"], note["date_basis"], note["claim"])
            row = merged.setdefault(key, {"date": key[0], "role": key[1], "date_basis": key[2],
                                           "event": key[3], "evidence": []})
            item = {"source_url": source["url"], "quote": note["quote"], "source_date": note["source_date"]}
            if item not in row["evidence"]:
                row["evidence"].append(item)
    return sorted(merged.values(), key=lambda r: r["date"])


async def group_periods(model, rows, language, question=""):
    dated = [r for r in rows if r["date"] != "unknown"]
    view = {"groups": [], "undated": [r for r in rows if r["date"] == "unknown"]}
    if not dated:
        return view
    compact = [{"number": i + 1, "date": r["date"], "finding": r["event"], "role": r["role"]} for i, r in enumerate(dated)]
    result = await model.generate(
        'Create a chronological digest of the substantive developments. Identify repeated descriptions of one event. '
        'Within each period, list the representative one-based entry numbers for significant developments, leaving out '
        'tangential details and duplicate claims. Keep genuinely different milestones with their own evidence. '
        'Only pick developments that help answer the question. If an entire period is irrelevant, use omit=true. '
        'The period summary must discuss only the selected representative entries. '
        'End each period at a strictly increasing entry number. The last period should cover the last dated item. '
        'Write in the supplied language. Return JSON {"periods":[{"through":last_entry_number,"keep":[entry_numbers],'
        '"title":"...","summary":"main supported conclusion"}]}. Use periods of unequal length as appropriate. '
        'Differentiate planned outcomes from past events.',
        {"question":question, "language": language, "entries": compact}, structured=True)
    start = 0
    for period in result.get("periods", []):
        if not isinstance(period, dict):
            continue
        end = period.get("through")
        if not isinstance(end, int) or not start < end <= len(dated):
            continue
        block = dated[start:end]
        if period.get("omit") is True:
            start = end
            continue
        chosen = sorted({n for n in (period.get("keep") or []) if type(n) is int and start < n <= end})
        entries = [dated[n-1] for n in chosen] or block
        view["groups"].append({"title": period.get("title") or entries[0]["event"], "summary": period.get("summary") or "",
                               "start": entries[0]["date"], "end": entries[-1]["date"], "events": entries})
        start = end
    if start < len(dated):
        block = dated[start:]
        view["groups"].append({"start": block[0]["date"], "end": block[-1]["date"], "title": block[0]["event"], "events": block})
    return view


async def daily_digest(model, request, rows):
    """Select dated developments using verified observations as the only candidate set."""
    by_day = {}
    for r in rows:
        if (len(r["date"]) == 10 and r["date"] <= (request.get("end_date") or "9999-12-31")
                and r["date_basis"] == "event" and r["role"] in {"observed","announced"}):
            by_day.setdefault(r["date"], []).append(r["event"])
    if not by_day:
        return []
    result = await model.generate(
        'Prepare a news event timeline answering the question. Use only candidate dates and findings provided. '
        'Focus on developments central to the topic. Deduplicate paraphrases of the same development. For each candidate day '
        'write concise, self-contained news sentences with actor, action and outcome. Shared actors or adjacent dates alone '
        'do not establish a connection. Exclude unrelated engagements, mere mention dates and projected events. '
        'Account for each candidate day explicitly. For days with no relevant development, put the date in discard. '
        'Return JSON {"days":[{"date":"YYYY-MM-DD","events":["..."]}],"discard":["YYYY-MM-DD"]}.',
        {"question":request["inputs"],"language":request.get("lang","en"),"candidates":by_day}, structured=True)
    chosen = {}
    for entry in result.get("days", []):
        if not isinstance(entry, dict) or entry.get("date") not in by_day:
            continue
        events = [e.strip() for e in entry.get("events",[]) if isinstance(e,str) and e.strip()]
        if events:
            chosen[entry["date"]] = list(dict.fromkeys(events))
    # Evidence determines date coverage. Editorial compression can improve wording,
    # but a skipped array entry should not erase an already verified day.
    discarded = {d for d in result.get("discard",[]) if isinstance(d,str) and d in by_day}
    return [{"start":day, "events":chosen.get(day) or [events[0]]} for day,events in sorted(by_day.items())
            if day in chosen or day not in discarded]
