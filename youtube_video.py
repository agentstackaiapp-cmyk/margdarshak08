"""
youtube_video.py
────────────────
Uses yt-dlp to search YouTube for real video IDs without an API key.
yt-dlp handles all of YouTube's anti-bot / consent-wall logic automatically.

Install once:  pip install yt-dlp
"""

import asyncio
from typing import Optional
import yt_dlp

# ── in-memory cache: query → video_id ─────────────────────────────────────
_cache: dict[str, Optional[str]] = {}

_YDL_OPTS: dict = {
    "quiet":        True,
    "no_warnings":  True,
    "extract_flat": True,    # don't download, just get metadata
    "skip_download": True,
    "default_search": "ytsearch",
}


def _search_sync(query: str) -> Optional[str]:
    """Blocking yt-dlp call — always run via run_in_executor."""
    try:
        with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            entries = (info or {}).get("entries") or []
            if entries:
                return entries[0].get("id")
    except Exception as exc:
        print(f"[youtube_video] yt-dlp error for '{query}': {exc}")
    return None


async def get_video_id(query: str) -> Optional[str]:
    """Async wrapper — returns the first real YouTube video ID or None."""
    if query in _cache:
        return _cache[query]

    loop = asyncio.get_event_loop()
    video_id = await loop.run_in_executor(None, _search_sync, query)
    _cache[query] = video_id
    return video_id


def embed_url(video_id: str) -> str:
    return (
        f"https://www.youtube.com/embed/{video_id}"
        "?autoplay=1&rel=0&modestbranding=1&playsinline=1"
    )


async def get_embed_url(query: str) -> Optional[str]:
    """Convenience: search + build embed URL in one call."""
    vid = await get_video_id(query)
    return embed_url(vid) if vid else None
