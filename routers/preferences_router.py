"""
routers/preferences_router.py
──────────────────────────────
FastAPI router for user preference endpoints.

GET  /api/preferences          — fetch current user's preferences
POST /api/preferences          — save / update preferences
GET  /api/preferences/schema   — return the onboarding question schema (drives the frontend)
"""

from fastapi import APIRouter, Request, Header
from typing import Optional

from models.preferences import PreferencesUpdate
from services.user_preferences_service import get_preferences, save_preferences

# Import get_current_user from the root server module at runtime to avoid
# circular imports — FastAPI resolves Depends lazily so this is safe.
import importlib

router = APIRouter(prefix="/api")


def _auth():
    """Lazy import of get_current_user from server to avoid circular import."""
    return importlib.import_module("server").get_current_user


# ── Onboarding question schema ────────────────────────────────────────────────
ONBOARDING_SCHEMA = [
    {
        "id": "deity",
        "step": 1,
        "emoji": "🙏",
        "title": "Who is your Ishta Devata?",
        "subtitle": "The divine form you feel closest to",
        "multi_select": False,
        "options": [
            {"key": "krishna",  "label": "Krishna",      "subtitle": "The Divine Guide",   "emoji": "🦚", "color": "#F59E0B"},
            {"key": "mahadev",  "label": "Mahadev",      "subtitle": "Lord of Yogis",      "emoji": "🔱", "color": "#8B5CF6"},
            {"key": "rama",     "label": "Shri Rama",    "subtitle": "Ideal of Dharma",    "emoji": "🏹", "color": "#10B981"},
            {"key": "devi",     "label": "Devi / Shakti","subtitle": "Divine Mother",      "emoji": "🌸", "color": "#F472B6"},
            {"key": "all",      "label": "All forms",    "subtitle": "Ekam sat, many paths","emoji": "🕉️","color": "#F97316"},
        ],
    },
    {
        "id": "scriptures",
        "step": 2,
        "emoji": "📖",
        "title": "Which scriptures speak to your soul?",
        "subtitle": "Select all that resonate with you",
        "multi_select": True,
        "options": [
            {"key": "gita",        "label": "Bhagavad Gita",    "subtitle": "Krishna's wisdom",   "emoji": "📖", "color": "#F97316"},
            {"key": "upanishads",  "label": "Upanishads",       "subtitle": "Atman & Brahman",    "emoji": "🌸", "color": "#8B5CF6"},
            {"key": "ramayana",    "label": "Ramayana",         "subtitle": "Rama's journey",     "emoji": "🏹", "color": "#10B981"},
            {"key": "mahabharata", "label": "Mahabharata",      "subtitle": "Epic of dharma",     "emoji": "⚔️", "color": "#6366F1"},
            {"key": "vedas",       "label": "Vedas & Puranas",  "subtitle": "Ancient wisdom",     "emoji": "🕉️", "color": "#F59E0B"},
            {"key": "all",         "label": "All scriptures",   "subtitle": "The whole ocean",    "emoji": "✨", "color": "#F97316"},
        ],
    },
    {
        "id": "goals",
        "step": 3,
        "emoji": "🌟",
        "title": "What does your soul seek?",
        "subtitle": "Choose all that call to you",
        "multi_select": True,
        "options": [
            {"key": "peace",  "label": "Inner Peace",  "subtitle": "Shanti & calm",      "emoji": "🧘", "color": "#38BDF8"},
            {"key": "wisdom", "label": "Wisdom",       "subtitle": "Jnana & clarity",    "emoji": "💡", "color": "#FBBF24"},
            {"key": "bhakti", "label": "Devotion",     "subtitle": "Bhakti & love",      "emoji": "❤️", "color": "#F472B6"},
            {"key": "karma",  "label": "Right Action", "subtitle": "Karma Yoga",         "emoji": "⚖️", "color": "#34D399"},
            {"key": "moksha", "label": "Liberation",   "subtitle": "Moksha & freedom",   "emoji": "🌟", "color": "#A78BFA"},
        ],
    },
    {
        "id": "language",
        "step": 4,
        "emoji": "💬",
        "title": "How should I speak to you?",
        "subtitle": "Your preferred language for guidance",
        "multi_select": False,
        "options": [
            {"key": "hindi",    "label": "हिन्दी",       "subtitle": "Hindi only",          "emoji": "🇮🇳", "color": "#F97316"},
            {"key": "english",  "label": "English",      "subtitle": "English only",        "emoji": "🌍", "color": "#38BDF8"},
            {"key": "hinglish", "label": "Hinglish",     "subtitle": "Hindi + English mix", "emoji": "🔄", "color": "#10B981"},
            {"key": "sanskrit", "label": "Sanskrit-rich","subtitle": "Original shlokas",    "emoji": "🪔", "color": "#FBBF24"},
        ],
    },
]


@router.get("/preferences/schema")
async def get_schema():
    """Return the onboarding question schema. No auth required."""
    return {"questions": ONBOARDING_SCHEMA, "total_steps": len(ONBOARDING_SCHEMA)}


@router.get("/preferences")
async def get_user_preferences(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    user = await _auth()(request, authorization)
    prefs = await get_preferences(user.user_id)
    if not prefs:
        return {"onboarding_completed": False, "deities": [], "scriptures": [],
                "spiritual_goals": [], "language_pref": "hinglish"}
    return prefs.model_dump(exclude={"created_at", "updated_at"})


@router.post("/preferences")
async def update_user_preferences(
    payload: PreferencesUpdate,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    user = await _auth()(request, authorization)
    saved = await save_preferences(user.user_id, payload)
    return saved.model_dump(exclude={"created_at", "updated_at"})
