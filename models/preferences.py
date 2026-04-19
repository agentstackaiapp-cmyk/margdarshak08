"""
models/preferences.py
─────────────────────
Pydantic models for user onboarding preferences.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


class UserPreferences(BaseModel):
    user_id: str

    # Q1 — Deity / Ishta Devata (single select, but stored as list for flexibility)
    deities: List[str] = []         # krishna | mahadev | rama | devi | all

    # Q2 — Preferred scriptures (multi-select)
    scriptures: List[str] = []      # gita | upanishads | ramayana | mahabharata | vedas | all

    # Q3 — Spiritual goals (multi-select)
    spiritual_goals: List[str] = [] # peace | wisdom | bhakti | karma | moksha

    # Q4 — Language preference (single select)
    language_pref: str = "hinglish" # hindi | english | hinglish | sanskrit

    onboarding_completed: bool = False

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PreferencesUpdate(BaseModel):
    deities: Optional[List[str]] = None
    scriptures: Optional[List[str]] = None
    spiritual_goals: Optional[List[str]] = None
    language_pref: Optional[str] = None
    onboarding_completed: Optional[bool] = None
