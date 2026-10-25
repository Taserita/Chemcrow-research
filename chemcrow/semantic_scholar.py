"""Process-shared, paced metadata requests for ScientificResearch only."""

import atexit
import copy
import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key, interval=1.2, max_attempts=5, wait_budget=120,
                 cache_ttl=3600):
        if not api_key:
            raise ValueError("SEMANTIC_SCHOLAR_API_KEY is required for online research.")
        if not math.isfinite(interval) or interval < 1 or max_attempts < 1:
            raise ValueError("interval must be at least 1 second; max_attempts must be positive.")
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key})
        self.interval = interval
        self.max_attempts = max_attempts
        self.wait_budget = wait_budget
        self.cache_ttl = cache_ttl
        self.next_allowed_at = 0.0
        self.lock = threading.Lock()
        self.cache = {}

    @staticmethod
    def _retry_delay(value, attempt):
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = (date - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = float("nan")
            if math.isfinite(delay):
                return max(0, delay)
        return min(60, 5 * 2 ** attempt) + random.uniform(0, 1)

    def get(self, endpoint, params=None):
        # Cache identity includes pagination and requested fields, never the key.
        cache_key = (endpoint, json.dumps(params or {}, sort_keys=True))
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                return copy.deepcopy(cached[1])
            waited = 0.0
            for attempt in range(self.max_attempts):
                wait = max(0, self.next_allowed_at - time.monotonic())
                if waited + wait > self.wait_budget:
                    raise RuntimeError("Semantic Scholar cooldown exceeds the request wait budget.")
                time.sleep(wait)
                waited += wait
                self.next_allowed_at = time.monotonic() + self.interval
                response = None
                try:
                    response = self.session.get(
                        f"{self.BASE_URL}/{endpoint.lstrip('/')}",
                        params=params, timeout=30,
                    )
                    if response.status_code in (429, 502, 503, 504):
                        delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
                        # Persist even on the final failed attempt, for the next call.
                        self.next_allowed_at = max(self.next_allowed_at, time.monotonic() + delay)
                        if attempt + 1 == self.max_attempts:
                            raise RuntimeError(f"Semantic Scholar HTTP {response.status_code}; retry budget exhausted.")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    self.cache[cache_key] = (time.monotonic() + self.cache_ttl, copy.deepcopy(data))
                    # Bound memory use in a long-running agent.
                    if len(self.cache) > 128:
                        self.cache.pop(next(iter(self.cache)))
                    return data
                except (requests.Timeout, requests.ConnectionError):
                    self.next_allowed_at = max(
                        self.next_allowed_at, time.monotonic() + self._retry_delay(None, attempt)
                    )
                    if attempt + 1 == self.max_attempts:
                        raise RuntimeError("Semantic Scholar connection retry budget exhausted.") from None
                finally:
                    if response is not None:
                        response.close()

    def search_candidates(self, query, limit=10, offset=0):
        return self.get("paper/search", params={
            "query": query, "limit": limit, "offset": offset,
            "fields": "title,abstract,year,url,externalIds,openAccessPdf,isOpenAccess,citationStyles",
        })

    def close(self):
        with self.lock:
            self.session.close()


_clients = {}
_clients_lock = threading.Lock()


def get_client(api_key):
    """All ScientificResearch calls using the same key share pacing and cache."""
    with _clients_lock:
        if api_key not in _clients:
            _clients[api_key] = SemanticScholarClient(api_key)
        return _clients[api_key]


@atexit.register
def _close_clients():
    for client in _clients.values():
        client.close()
