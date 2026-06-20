"""In-memory TTL cache — avoids hitting CMC rate limits on repeated calls."""

import time
import threading
from typing import Any


class TTLCache:
    """Thread-safe dict with per-key expiry."""

    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return default
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return default
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Global singleton
cache = TTLCache()
