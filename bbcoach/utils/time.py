from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def days_until(target: Optional[date], today: Optional[date] = None) -> Optional[int]:
    if target is None:
        return None
    today = today or date.today()
    return (target - today).days
