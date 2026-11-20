"""Read one published monthly level series and derive consecutive net changes."""

import re
from datetime import date
from decimal import Decimal
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .monthly_series import shift_month

FRED_HOST = "fred.stlouisfed.org"
ROW = re.compile(
    r"(?m)^\|[ \t]*(\d{4}-\d{2}-01)[ \t]*\|[ \t]*([+-]?[\d,.]+)[ \t]*\|[ \t]*$"
    r"|(?<!\S)#(\d{4}-\d{2}-01)\|([+-]?[\d,.]+)(?=\s|$)"
)
PEOPLE = {"人", "人数", "个岗位", "people", "persons", "jobs", "employees"}


def fred_path(url):
    parts = urlsplit(url)
    if parts.netloc.lower() == FRED_HOST and parts.scheme == "https" and not parts.query:
        return re.fullmatch(r"/(series|data)/([A-Za-z0-9_]+)/?", parts.path)
    return None


def history_link(document):
    path = fred_path(document.url)
    if not path or path[1] != "series":
        return None
    for link in re.findall(r"\[[^\]\n]+\]\(([^()\s]+)\)", document.content):
        url = urljoin(document.url, link)
        target = fred_path(url)
        if target and target[1].lower() == "data" and target[2].lower() == path[2].lower():
            return url
    return None


def field(text, name):
    return re.search(rf"(?m)^\|[ \t]*{name}[ \t]*\|[ \t]*([^|\n]+)\|[ \t]*$", text)


def terms_match(text, terms, aliases=None):
    def words(value):
        return {
            aliases.get(word, word) if aliases else word
            for word in re.findall(r"\w+", value.casefold())
        }

    wanted = words(" ".join(terms))
    return bool(wanted) and wanted <= words(text)


def title_matches(text, terms, unit):
    aliases = (
        {word: "employment" for word in ("employee", "employees", "payroll", "payrolls")}
        if unit.strip().casefold() in PEOPLE
        else None
    )
    return terms_match(text, terms, aliases=aliases)


def normalize_title_terms(terms, adjustment):
    title = []
    for term in terms:
        modifier = re.search(r"\b(?:not )?seasonally adjusted\b", term, re.I)
        if modifier:
            stated = modifier[0].lower().replace(" ", "_")
            if adjustment not in {"any", stated}:
                raise ValueError("标题检索词与所选季调口径冲突")
            adjustment = stated
            term = term[: modifier.start()] + term[modifier.end() :]
        if term := re.sub(r"\bmonthly\b|\blevels?\b", "", term, flags=re.I).strip():
            title.append(term)
    if not title:
        raise ValueError("指标标题需包含口径以外的区分词")
    return title, adjustment


class RepositorySeriesLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.links = {}

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        url = urljoin(f"https://{FRED_HOST}/", dict(attrs).get("href") or "")
        match = fred_path(url)
        self.current = (url, []) if match and match[1] == "series" else None

    def handle_data(self, data):
        if self.current:
            self.current[1].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current:
            url, chunks = self.current
            title = " ".join("".join(chunks).split())
            if title:
                self.links.setdefault(url, title)
            self.current = None


def publisher_matches(text, terms):
    words = [
        word
        for word in re.findall(r"[A-Za-z]+", text)
        if word.lower() not in {"a", "an", "and", "of", "the", "for"}
    ]
    initials = "".join(word[0].upper() for word in words)
    return all(
        terms_match(text, [term])
        or (re.fullmatch(r"[A-Z]{2,8}", term) is not None and term in initials)
        for term in terms
    )


def series_metadata(text, unit):
    names = (
        "Title",
        "Series ID",
        "Source",
        "Seasonal Adjustment",
        "Frequency",
        "Units",
        "Date Range",
    )
    matches = {name: field(text, name) for name in names}
    if not all(matches.values()):
        raise ValueError("原文缺少序列标题、发布者、频率、单位或日期范围")
    values = {name: match[1].strip() for name, match in matches.items()}
    if values["Frequency"].casefold() != "monthly":
        raise ValueError("原表频率不是月度")
    source_unit = values["Units"].casefold()
    scale = 1000 if source_unit.startswith("thousands of ") else 1
    base_unit = source_unit.removeprefix("thousands of ")
    asked_unit = unit.strip().casefold()
    if not (asked_unit == base_unit or (asked_unit in PEOPLE and base_unit in PEOPLE)):
        raise ValueError("原表单位无法转换为计划指定的单位")
    return (
        values,
        scale,
        (min(m.start() for m in matches.values()), max(m.end() for m in matches.values())),
    )


def levels(text):
    observations = {}
    for row in ROW.finditer(text):
        day, number = row.group(1, 2) if row[1] else row.group(3, 4)
        date.fromisoformat(day)
        value = Decimal(number.replace(",", ""))
        month = day[:7]
        entry = observations.get(month)
        if entry and entry[0] != value:
            raise ValueError(f"{month} 在同一原表中有冲突值")
        observations[month] = (value, row.span())
    return observations


def net_change(previous, current, scale):
    value = (current - previous) * scale
    if value != value.to_integral_value() or abs(value) > 10**15:
        raise ValueError("环比差值超出当前图表可表示的整数范围")
    return int(value)


def extract_monthly_changes(document, spec, title_terms, publisher_terms, adjustment, cutoff=None):
    path = fred_path(document.url)
    if not path or path[1] != "data":
        raise ValueError("链接不是支持的历史数据表")
    metadata, scale, metadata_span = series_metadata(document.content, spec["unit"])
    if metadata["Series ID"].casefold() != path[2].casefold():
        raise ValueError("链接与原表的序列编号不一致")
    if not title_matches(metadata["Title"], title_terms, spec["unit"]) or not publisher_matches(
        metadata["Source"], publisher_terms
    ):
        raise ValueError("原表指标或原始发布者与计划不一致")
    if adjustment != "any" and metadata["Seasonal Adjustment"].casefold() != adjustment.replace(
        "_", " "
    ):
        raise ValueError("原表季调口径与计划不一致")
    observations = levels(document.content)
    match = re.fullmatch(r"\d{4}-\d{2}-01 to (\d{4}-\d{2}-01)", metadata["Date Range"])
    latest = match[1][:7] if match else None
    if not latest or max(observations, default="") != latest:
        raise ValueError("原表没有完整读到元数据标明的最新观察月")
    if (cutoff and latest > cutoff[:7]) or (
        document.retrieved_at and latest >= document.retrieved_at.strftime("%Y-%m")
    ):
        raise ValueError("原表含有超出调查范围或尚未结束的观察月")
    if document.retrieved_at and latest < shift_month(document.retrieved_at.strftime("%Y-%m"), -3):
        raise ValueError("序列末期较早，需查明最新已发布周期")
    months = [shift_month(latest, offset) for offset in range(-spec["periods"], 1)]
    missing = [month for month in months if month not in observations]
    if missing:
        raise ValueError("原表缺少用于环比计算的相邻月水平：" + "、".join(missing))
    points = []
    for previous, current in zip(months[:-1], months[1:], strict=True):
        old, old_span = observations[previous]
        new, new_span = observations[current]
        points.append(
            {
                "month": current,
                "value": net_change(old, new, scale),
                "spans": [old_span, new_span, metadata_span],
            }
        )
    return metadata, points


def calculated_value_in_original(value, month, references, unit, documents):
    if len({r["version_id"] for r in references}) != 1 or len(references) != 3:
        return False
    path = fred_path(documents[references[0]["version_id"]].url)
    if not path or path[1] != "data":
        return False
    try:
        *rows, metadata = (ref["quote"] for ref in references)
        fields, scale, _ = series_metadata(metadata, unit)
        previous = shift_month(month, -1)
        observations = levels("\n".join(rows))
        return (
            fields["Series ID"].casefold() == path[2].casefold()
            and set(observations) == {previous, month}
            and net_change(observations[previous][0], observations[month][0], scale) == value
        )
    except (ValueError, ArithmeticError):
        return False
