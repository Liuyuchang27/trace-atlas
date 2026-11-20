"""Fetch an approved first-party HTTPS page directly for freshness checks."""

import asyncio
import socket
from ipaddress import ip_address
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .retrieval import RetrievalFailure, public_url

MAX_HTML_BYTES = 3 * 1024**2


async def _public_addresses(host):
    records = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses = sorted({r[4][0] for r in records})
    if not addresses or any(not ip_address(value).is_global for value in addresses):
        raise RetrievalFailure("origin_nonpublic_address")
    return addresses


async def fetch_origin_html(url, domains, timeout):
    """Pin the connection to a checked public address and verify TLS for the hostname."""
    async with asyncio.timeout(timeout):
        for _ in range(3):
            parts = httpx.URL(public_url(url))
            host = parts.host
            if (
                parts.scheme != "https"
                or parts.port not in (None, 443)
                or not any(host == d or host.endswith("." + d) for d in domains)
            ):
                raise RetrievalFailure("origin_outside_official_domains")
            addresses = await _public_addresses(host)
            # Public IPv4 is preferred on hosts without an IPv6 route.
            address = min(addresses, key=lambda value: (":" in value, value))
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                async with client.stream(
                    "GET",
                    parts.copy_with(host=address),
                    headers={
                        "Host": host,
                        "User-Agent": "Mozilla/5.0 TraceAtlas",
                        "Accept": "text/html,application/xhtml+xml",
                        "Cache-Control": "no-cache",
                    },
                    extensions={"sni_hostname": host},
                ) as response:
                    if response.is_redirect:
                        if not (location := response.headers.get("location")):
                            raise RetrievalFailure("origin_missing_redirect_target")
                        url = urljoin(str(parts), location)
                        continue
                    response.raise_for_status()
                    if "html" not in response.headers.get("content-type", "").lower():
                        raise RetrievalFailure("origin_not_html")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_HTML_BYTES:
                            raise RetrievalFailure("origin_page_too_large")
                    return str(parts), bytes(body)
        raise RetrievalFailure("origin_redirect_limit")


def page_text(html):
    """Preserve visible text, dates and exact href targets in a text extraction."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup.select("script, style, noscript, template, svg, img, iframe"):
        tag.decompose()
    root = soup.find("main") or soup.find("body") or soup
    for link in root.find_all("a", href=True):
        label = " ".join(link.stripped_strings)
        href = link["href"]
        if label and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            link.replace_with(f"[{label}]({href})")
    for tag in root.find_all(["h1", "h2", "h3", "p", "time", "li"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = "\n\n".join(
        line.strip() for line in root.get_text(" ", strip=False).splitlines() if line.strip()
    )
    return title, text
