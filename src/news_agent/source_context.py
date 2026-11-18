"""Source and page-date roles derived only from selected verbatim text.

Labels describe what a page says, not verified authorship or truth. Bare 时间 is
only a page header when anchored on a source-labelled line; event prose is preserved.
"""

import calendar
import re
from urllib.parse import urlsplit

from .answer_contract import dates

PUBLICATION = re.compile(
    r"发布时间|发布日期|发表时间|发布于|更新时间|转载时间|"
    r"\bpublished(?:\s+(?:on|at))?\b|\bupdated(?:\s+(?:on|at))?\b|"
    r"(?m:^[ \t]*时间(?=[：:][^\n]{0,120}来源[：:]))",
    re.I,
)


def publication_mentions(text):
    """Extract only the date attached to a publication label, not other event prose."""
    found = []
    month_numbers = {
        name.lower(): m
        for m in range(1, 13)
        for name in [calendar.month_name[m], calendar.month_abbr[m]]
    }
    months = "|".join(sorted(month_numbers, key=len, reverse=True))
    for label in PUBLICATION.finditer(text):
        tail = text[label.end() :]
        full = next((d for d in dates(tail) if re.fullmatch(r"[\s:：]*", tail[: d[1]])), None)
        if full:
            found.append(
                {
                    "start": label.start(),
                    "end": label.end() + full[2],
                    "value": full[0],
                    "precision": "day",
                }
            )
            continue
        month = re.match(
            rf"[\s:：]*(?:(\d{{4}})[-/](\d{{1,2}})(?![-/\d])|"
            rf"(\d{{4}})年(\d{{1,2}})月|({months})\.?\s+(\d{{4}}))",
            tail,
            re.I,
        )
        if month:
            y = month[1] or month[3] or month[6]
            m = int(month[2] or month[4] or month_numbers[month[5].lower()])
            if 1 <= m <= 12:
                found.append(
                    {
                        "start": label.start(),
                        "end": label.end() + month.end(),
                        "value": f"{y}-{m:02d}",
                        "precision": "month",
                    }
                )
                continue
        year = re.match(r"[\s:：]*(\d{4})(?!\d)", tail)
        if year:
            found.append(
                {
                    "start": label.start(),
                    "end": label.end() + year.end(),
                    "value": year[1],
                    "precision": "year",
                }
            )
    for item in found:
        label_text = text[item["start"] : item["end"]]
        item["role"] = (
            "page_updated"
            if re.match(r"更新时间|updated", label_text, re.I)
            else "reprint_time"
            if label_text.startswith("转载时间")
            else "page_timestamp"
            if label_text.strip().startswith("时间")
            else "publication_time"
        )
        item["original_text"] = label_text
    return found


def source_records(evidence, references):
    """Exact selected-reference metadata; never inspect omitted page text for facts."""
    records = []
    for ref in references:
        doc = evidence.documents[ref["version_id"]]
        quote = ref["quote"]
        origins = []
        for match in re.finditer(r"(?:^|\s)来源\s*[:：]\s*([^\n|；;]{1,100})", quote):
            origins.append(
                {
                    "name": match[1].strip(),
                    "original_text": match[0],
                    "start": ref["start"] + match.start(),
                    "end": ref["start"] + match.end(),
                }
            )
        records.append(
            {
                "version_id": ref["version_id"],
                "passage_id": ref["passage_id"],
                "hosting_domain": urlsplit(doc.url).hostname,
                "attributed_origins": origins,
                "origin_status": "attributed_only" if origins else "unknown",
                "primary_status": "not_established",
                "page_dates": [
                    {
                        **item,
                        "start": ref["start"] + item["start"],
                        "end": ref["start"] + item["end"],
                    }
                    for item in publication_mentions(quote)
                ],
            }
        )
    return records
