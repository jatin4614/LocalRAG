"""In-process state tracker for model sleep/wake + idle timestamps."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Literal, Union


State = Literal["awake", "asleep", "unknown"]


class IdleTracker:
    def __init__(self, now_fn: Callable[[], float] = time.monotonic) -> None:
        self._now = now_fn
        self._state: Dict[str, State] = {}
        self._last_active: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, *, initial_state: State = "asleep") -> None:
        with self._lock:
            self._state[name] = initial_state
            self._last_active[name] = self._now()

    def mark_awake(self, name: str) -> None:
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            self._state[name] = "awake"
            self._last_active[name] = self._now()

    def mark_asleep(self, name: str) -> None:
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            self._state[name] = "asleep"

    def touch(self, name: str) -> None:
        """Bump last_active only if awake; no-op if asleep."""
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            if self._state[name] == "awake":
                self._last_active[name] = self._now()

    def snapshot(self) -> Dict[str, Dict[str, Union[float, str]]]:
        with self._lock:
            return {
                n: {"state": self._state[n], "last_active": self._last_active[n]}
                for n in self._state
            }

    def idle_models(self, *, idle_secs: float) -> List[str]:
        now = self._now()
        with self._lock:
            return [
                n for n, s in self._state.items()
                if s == "awake" and (now - self._last_active[n]) >= idle_secs
            ]
