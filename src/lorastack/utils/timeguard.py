"""TimeGuard: hard session deadline (Kaggle 12h). At trigger: save ckpt + collect + stop clean."""
from __future__ import annotations

import time


class TimeGuard:
    def __init__(self, budget_minutes: int, warn_fraction: float = 1.0):
        self.deadline = time.time() + budget_minutes * 60
        self.reason: str | None = None

    def expired(self) -> bool:
        if time.time() >= self.deadline:
            self.reason = "timeguard"
            return True
        return False

    def remaining_minutes(self) -> float:
        return (self.deadline - time.time()) / 60
