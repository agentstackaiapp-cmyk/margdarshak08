"""
db.py — Shared MongoDB singleton.

Usage:
    import db
    db.init(mongo_url, db_name)     # once at startup (server.py)
    col = db.get().my_collection    # anywhere else
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def init(mongo_url: str, db_name: str) -> AsyncIOMotorDatabase:
    global _client, _db
    _client = AsyncIOMotorClient(mongo_url)
    _db = _client[db_name]
    return _db


def get() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised — call db.init() first.")
    return _db
