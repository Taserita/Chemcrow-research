"""Validated PDF retrieval for ScientificResearch, independent of legacy search."""

import asyncio
import math
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import aiohttp
import fitz
from paperscraper import lib
from pypdf import PdfReader

from chemcrow.paperscraper_compat import find_pmc_pdf_link


MAX_BYTES = 50 * 1024 * 1024
ROUTE_TIMEOUT = 60
PAPER_TIMEOUT = 180


def validate_pdf(path):
    """Require a real PDF readable by PaperQA's parser, with extractable text."""
    path = Path(path)
    if not path.exists():
        return {"status": "missing_file"}
    with path.open("rb") as stream:
        prefix = stream.read(1024).lstrip()
    if not prefix.startswith(b"%PDF-"):
        html = b"<html" in prefix.lower() or b"<!doctype html" in prefix.lower()
        return {"status": "html_instead_of_pdf" if html else "not_pdf", "bytes": path.stat().st_size}
    try:
        with fitz.open(path) as document:
            if not document.is_pdf:
                return {"status": "not_pdf"}
            if document.needs_pass:
                return {"status": "encrypted_pdf"}
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            return {"status": "encrypted_pdf"}
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
        result = {"pages": len(texts), "text_pages": sum(bool(t) for t in texts),
                  "text_characters": sum(len(t) for t in texts), "bytes": path.stat().st_size}
        result["status"] = "valid_pdf" if any(texts) else "no_text"
        return result
    except Exception as exc:
        return {"status": "invalid_pdf", "error": type(exc).__name__}


class DownloadResponse:
    """Small buffered response compatible with paper-scraper's read-only usage."""
    def __init__(self, url, status, headers, body, request_info):
        self.url, self.status, self.headers = url, status, headers
        self.body, self.request_info = body, request_info
        self.ok = status < 400

    async def read(self):
        return self.body

    async def text(self):
        return self.body.decode("utf-8")

    def raise_for_status(self):
        if not self.ok:
            raise aiohttp.ClientResponseError(self.request_info, (), status=self.status)


class DownloadSession:
    """Audit requests and reuse responses across routes for one paper.

    HTTP redirects are followed explicitly so fallback routes cannot request an
    already visited final URL again. The underlying session has no API key.
    """
    def __init__(self, session, next_request):
        self.session = session
        self.next_request = next_request
        self.cache = {}
        self.failures = {}
        self.requests = []

    @asynccontextmanager
    async def get(self, url, **kwargs):
        yield await self.fetch(str(url))

    async def fetch(self, url):
        aliases = []
        for _ in range(10):
            url = urldefrag(url)[0]
            parsed = urlparse(url)
            if parsed.scheme not in ("https", "http") or not parsed.netloc:
                raise ValueError("Invalid HTTP download URL")
            if url in self.failures:
                self.requests.append({"url": url, "cached": True, "error": self.failures[url]})
                raise RuntimeError("A previous request to this URL already failed")
            if url in self.cache:
                response = self.cache[url]
                self.requests.append({"url": url, "final_url": str(response.url),
                                      "http_status": response.status, "cached": True})
                for alias in aliases:
                    self.cache[alias] = response
                return response
            if url in aliases:
                raise ValueError("Redirect loop")
            aliases.append(url)
            for attempt in range(2):
                host = parsed.netloc.lower()
                await asyncio.sleep(max(0, self.next_request.get(host, 0) - time.monotonic()))
                self.next_request[host] = time.monotonic() + 1.0
                entry = {"url": url, "attempt": attempt + 1, "cached": False}
                self.requests.append(entry)
                try:
                    async with self.session.get(url, allow_redirects=False) as raw:
                        entry.update(http_status=raw.status, content_type=raw.headers.get("Content-Type", ""))
                        if raw.status in (301, 302, 303, 307, 308) and raw.headers.get("Location"):
                            url = urljoin(str(raw.url), raw.headers["Location"])
                            entry["redirect_to"] = url
                            break
                        body = bytearray()
                        async for chunk in raw.content.iter_chunked(65536):
                            body.extend(chunk)
                            if len(body) > MAX_BYTES:
                                raise ValueError("Download exceeds 50 MiB")
                        entry["bytes"] = len(body)
                        response = DownloadResponse(raw.url, raw.status, raw.headers, bytes(body), raw.request_info)
                        if raw.status in (429, 502, 503, 504):
                            delay = 2.0
                            retry = raw.headers.get("Retry-After")
                            if retry:
                                try:
                                    delay = max(0, float(retry))
                                except ValueError:
                                    try:
                                        date = parsedate_to_datetime(retry)
                                        if date.tzinfo is None:
                                            date = date.replace(tzinfo=timezone.utc)
                                        delay = max(0, (date - datetime.now(timezone.utc)).total_seconds())
                                    except (ValueError, TypeError, OverflowError):
                                        pass
                            if not math.isfinite(delay):
                                delay = 2.0
                            self.next_request[host] = max(self.next_request[host], time.monotonic() + delay)
                            if attempt == 0:
                                continue
                        for alias in aliases:
                            self.cache[alias] = response
                        return response
                except asyncio.CancelledError:
                    entry["error"] = "route_timeout_or_cancelled"
                    self.failures[url] = entry["error"]
                    raise
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                    entry["error"] = type(exc).__name__
                    if attempt == 1 or isinstance(exc, ValueError):
                        self.failures[url] = type(exc).__name__
                        raise
            else:
                raise RuntimeError("Download request failed")
            # Only redirects reach this point.
        raise ValueError("Too many redirects")


async def _direct(url, path, session):
    async with session.get(url) as response:
        response.raise_for_status()
        path.write_bytes(await response.read())
    return True


async def _openaccess(paper, path, session):
    """Resolve landing-page PDF links against the final URL after redirects."""
    url = (paper.get("openAccessPdf") or {}).get("url")
    if not url:
        return False
    async with session.get(url) as response:
        response.raise_for_status()
        if "pdf" in response.headers.get("Content-Type", "").lower():
            path.write_bytes(await response.read())
            return True
        html = await response.text()
        final_url = str(response.url)

    # Preserve paper-scraper's ChemRxiv and PDF-link discovery behavior, but
    # resolve all relative forms using the publisher URL rather than the DOI.
    chemrxiv = re.search(
        r'content="(https://chemrxiv.org/engage/api-gateway/chemrxiv/assets.*\.pdf)"',
        html,
    )
    if chemrxiv:
        pdf_link = chemrxiv.group(1)
    else:
        try:
            pdf_link = lib.search_pdf_link(html, epdf=True)
        except lib.NoPDFLinkError:
            pdf_link = lib.search_pdf_link(html)
    return await _direct(urljoin(final_url, pdf_link), path, session)


async def _pmc(paper, path, session):
    pmcid = (paper.get("externalIds") or {}).get("PubMedCentral")
    if not pmcid:
        return False
    url = await find_pmc_pdf_link(str(pmcid).removeprefix("PMC"), session)
    return await _direct(url, path, session)


async def _pubmed(paper, path, session):
    pmid = (paper.get("externalIds") or {}).get("PubMed")
    if not pmid:
        return False
    async with session.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/") as response:
        response.raise_for_status()
        match = re.search(r"PMC\d+", await response.text())
    if not match:
        raise ValueError("No PMC ID found on PubMed page")
    return await _pmc({"externalIds": {"PubMedCentral": match.group(0)}}, path, session)


def fallback_routes():
    # openaccess reuses the cached first response to discover a public PDF link
    # when the advertised URL is an HTML landing page instead of a direct PDF.
    return [("paperscraper_openaccess", _openaccess),
            ("paperscraper_arxiv", lib.arxiv_scraper),
            ("paperscraper_medrxiv", lib.medrxiv_scraper),
            ("paperscraper_biorxiv", lib.biorxiv_scraper),
            ("paperscraper_chemrxiv", lib.chemrxiv_scraper),
            ("pmc", _pmc), ("pubmed", _pubmed)]


async def download_paper(paper, path, record, http_session, next_request):
    """Return True only after atomically saving a readable PDF with text."""
    path = Path(path)
    temporary = path.with_suffix(".pdf.part")
    session = DownloadSession(http_session, next_request)
    attempts = record.setdefault("download_attempts", [])
    record["download_status"] = "attempted"
    started = time.monotonic()
    routes = []
    url = (paper.get("openAccessPdf") or {}).get("url")
    if url:
        async def direct_route(paper, path, session):
            return await _direct(url, path, session)
        routes.append(("open_access_pdf", direct_route))
    routes.extend(fallback_routes())
    try:
        for name, function in routes:
            remaining = PAPER_TIMEOUT - (time.monotonic() - started)
            if remaining <= 0:
                attempts.append({"method": name, "status": "skipped", "reason": "paper_timeout"})
                break
            temporary.unlink(missing_ok=True)
            entry = {"method": name, "status": "attempted", "requests": []}
            attempts.append(entry)
            session.requests = entry["requests"]
            try:
                ok = await asyncio.wait_for(function(paper, temporary, session), min(ROUTE_TIMEOUT, remaining))
                if not ok:
                    entry.update(status="skipped", reason="no_applicable_source")
                    continue
                validation = validate_pdf(temporary)
                entry["validation"] = validation
                if validation["status"] != "valid_pdf":
                    entry.update(status="failed", reason=validation["status"])
                    continue
                temporary.replace(path)
                entry["status"] = "completed"
                record.update(download_status="downloaded", local_path=str(path),
                              download_method=name, pdf_validation=validation)
                return True
            except Exception as exc:
                entry.update(status="failed", reason=type(exc).__name__)
        no_text = any(a.get("reason") == "no_text" for a in attempts)
        record.update(download_status="no_text" if no_text else "failed",
                      download_reason="PDF has no extractable text; OCR is not supported." if no_text else
                      "No readable PDF from available routes; see download_attempts.")
        return False
    finally:
        temporary.unlink(missing_ok=True)
