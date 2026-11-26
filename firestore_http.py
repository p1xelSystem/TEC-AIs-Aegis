import asyncio
import re
from urllib.parse import urlparse

import httpx

FIREBASE_CONSOLE_HOST = "console.firebase.google.com"
URL_RE = re.compile(r"https?://[^\s\"'<>)]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _keep_url(url):
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == FIREBASE_CONSOLE_HOST
        and parsed.path.startswith("/v1/r/project/")
    )


def _redact(text, secrets):
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 6:
            text = text.replace(secret, "[redacted]")
    text = URL_RE.sub(
        lambda m: m.group(0) if _keep_url(m.group(0)) else "[link removed]",
        text,
    )
    return EMAIL_RE.sub("[email removed]", text)


def error_detail(response, secrets=()):
    status = ""
    message = ""
    try:
        data = response.json()
        if isinstance(data, list) and data:
            data = data[0]
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            status = str(error.get("status", ""))
            message = str(error.get("message", ""))
    except ValueError:
        pass

    detail = "HTTP %d" % response.status_code
    if status:
        detail += " " + status
    if message:
        detail += ": " + message[:500]
    return _redact(detail, secrets)


def retry_after(response):
    value = response.headers.get("Retry-After", "").strip()
    return int(value) if value.isdigit() else 0


async def fetch(client, method, url, secrets=(), **kwargs):
    """Reads are retried on transient 5xx/network errors. Writes and quota
    errors (429) are never retried, so nothing is sent twice."""
    attempts = 3 if method.upper() == "GET" else 1
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError:
            if last:
                raise
            await asyncio.sleep(1 + attempt)
            continue
        if response.status_code >= 500 and not last:
            await asyncio.sleep(1 + attempt)
            continue
        return response
