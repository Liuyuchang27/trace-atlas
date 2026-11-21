"""Compute requested observation months and preserve source-backed numeric points."""

import re
from calendar import month_abbr, month_name
from decimal import Decimal
from itertools import groupby


def shift_month(month, offset):
    year, number = map(int, month.split("-"))
    year, index = divmod(year * 12 + number - 1 + offset, 12)
    return f"{year:04d}-{index + 1:02d}"


def reported_number(value, originals):
    value = Decimal(str(value))
    for quote in originals:
        for literal, scale in re.findall(
            r"(?<![\w.,])([+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\d|[.,]\d)"
            r"(?:\s*(thousand\b|[kK]\b|万))?",
            quote,
            flags=re.I,
        ):
            number = Decimal(literal.replace(",", "").replace("−", "-"))
            factor = {"万": 10000, "k": 1000, "thousand": 1000}.get(scale.lower(), 1)
            if number == value or number * factor == value:
                return True
    return False


def monthly_value_in_original(value, month, originals):
    from .event_graph import month_in_original

    if reported_number(value, originals):
        return True
    value = Decimal(str(value))
    for quote in originals:
        for is_table, block in groupby(
            quote.splitlines(), key=lambda line: line.lstrip().startswith("|")
        ):
            if not is_table:
                continue
            lines = list(block)
            if not any(
                re.search(r"\bin thousands\b|单位\s*[：:]?\s*千", row, re.I) for row in lines
            ):
                continue
            rows = [line.strip().strip("|").split("|") for line in lines]
            for i, header in enumerate(rows):
                matches = [
                    j for j, cell in enumerate(header[1:], 1) if month_in_original(month, cell)
                ]
                if len(matches) != 1:
                    continue
                col = matches[0]
                for row in rows[i + 1 :]:
                    if len(row) != len(header):
                        continue
                    cell = row[col].strip().replace(",", "").replace("−", "-")
                    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cell) and Decimal(cell) * 1000 == value:
                        return True
    return False


def explicit_search_months(query):
    months = set(re.findall(r"(?<!\d)\d{4}-(?:0[1-9]|1[0-2])(?!\d)", query))
    months.update(
        f"{year}-{int(month):02d}"
        for year, month in re.findall(r"(?<!\d)(\d{4})\s*年\s*(1[0-2]|0?[1-9])\s*月", query)
    )
    names = {
        name.lower(): number
        for number in range(1, 13)
        for name in (
            month_name[number],
            month_abbr[number],
            "Sept" if number == 9 else month_abbr[number],
        )
    }
    names_pattern = "|".join(sorted(names, key=len, reverse=True))
    for sequence, year in re.findall(
        rf"\b((?:(?:{names_pattern})\.?[\s,;/–-]+)+)(\d{{4}})\b", query, re.I
    ):
        for name in re.findall(rf"\b(?:{names_pattern})\b", sequence, re.I):
            months.add(f"{year}-{names[name.lower()]:02}")
    return months


def month_coverage(part, findings, *, allow_anchor_value=True):
    spec = part["monthly_series"]
    dated = [
        (row, f["event_time"]["start"], f)
        for row in findings
        if (f := row["finding"])["occurrence"] in {"observed", "statement"}
        and f["stance"] == "supports"
        and f["event_time"]["precision"] == "month"
        and row["references"]
    ]
    anchor = max(
        (
            month
            for _, month, f in dated
            if f.get("question_part_id") == spec["anchor_part_id"]
            and f.get("answers_question_part")
        ),
        default=None,
    )
    if not anchor:
        return {"anchor": None, "months": [], "missing": [], "points": {}}
    months = [shift_month(anchor, offset) for offset in range(1 - spec["periods"], 1)]
    points = {}
    for m in months:
        candidates = [
            (row["finding_id"], f.get("monthly_value"))
            for row, observed, f in dated
            if f["occurrence"] == "observed"
            and observed == m
            and (f.get("monthly_value") is not None or f.get("monthly_unavailable"))
            and (
                f.get("question_part_id") == part["part_id"]
                or (
                    allow_anchor_value
                    and m == anchor
                    and f.get("question_part_id") == spec["anchor_part_id"]
                )
            )
        ]
        if len({value for _, value in candidates}) == 1:
            fid, value = candidates[-1]
            points[m] = {
                "finding_id": fid,
                "value": value,
                **({"unavailable": True} if value is None else {}),
            }
    return {
        "anchor": anchor,
        "months": months,
        "missing": [m for m in months if m not in points],
        "points": points,
    }
