"""
services/user_preferences_service.py
─────────────────────────────────────
Pure DB operations for user preferences. No HTTP, no FastAPI — just data.
"""

from datetime import datetime, timezone
from typing import Optional

import db
from models.preferences import UserPreferences, PreferencesUpdate


async def get_preferences(user_id: str) -> Optional[UserPreferences]:
    doc = await db.get().user_preferences.find_one(
        {"user_id": user_id}, {"_id": 0}
    )
    return UserPreferences(**doc) if doc else None


async def save_preferences(
    user_id: str, update: PreferencesUpdate
) -> UserPreferences:
    """Upsert — creates on first call, merges on subsequent calls."""
    existing = await get_preferences(user_id)
    now = datetime.now(timezone.utc)

    if existing:
        patch = {k: v for k, v in update.model_dump().items() if v is not None}
        patch["updated_at"] = now
        await db.get().user_preferences.update_one(
            {"user_id": user_id}, {"$set": patch}
        )
    else:
        prefs = UserPreferences(
            user_id=user_id,
            deities=update.deities or [],
            scriptures=update.scriptures or [],
            spiritual_goals=update.spiritual_goals or [],
            language_pref=update.language_pref or "hinglish",
            onboarding_completed=update.onboarding_completed or False,
            created_at=now,
            updated_at=now,
        )
        await db.get().user_preferences.insert_one(prefs.model_dump())

    result = await get_preferences(user_id)
    return result  # type: ignore[return-value]
