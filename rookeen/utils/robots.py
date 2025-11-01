from __future__ import annotations

import time
from urllib.parse import urlparse

_last_visit: dict[str, float] = {}


def politeness_delay(url: str, rps: float = 0.5) -> None:
    host = urlparse(url).netloc
    now = time.time()
    last = _last_visit.get(host, 0.0)
    min_interval = 1.0 / max(rps, 0.01)
    if now - last < min_interval:
        time.sleep(min_interval - (now - last))
    _last_visit[host] = time.time()
