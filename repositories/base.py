"""
Repository Layer for Database Operations
FAANG-Level: Single Responsibility, testable, async patterns
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from core.exceptions import ResourceNotFoundError
from core.logging import get_logger

logger = get_logger(__name__)


class BaseRepository(ABC):
    """Base repository with common operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str):
        self.db = db
        self.collection = db[collection_name]
        self.collection_name = collection_name
    
    async def find_one(self, filter: Dict[str, Any], exclude_id: bool = True) -> Optional[Dict[str, Any]]:
        """Find a single document"""
        projection = {"_id": 0} if exclude_id else None
        doc = await self.collection.find_one(filter, projection)
        return doc
    
    async def find_many(
        self,
        filter: Dict[str, Any],
        limit: int = 100,
        skip: int = 0,
        sort: Optional[List[tuple]] = None,
        exclude_id: bool = True
    ) -> List[Dict[str, Any]]:
        """Find multiple documents with pagination"""
        projection = {"_id": 0} if exclude_id else None
        cursor = self.collection.find(filter, projection).limit(limit).skip(skip)
        
        if sort:
            cursor = cursor.sort(sort)
        
        docs = await cursor.to_list(length=limit)
        return docs
    
    async def insert_one(self, document: Dict[str, Any]) -> str:
        """Insert a single document"""
        result = await self.collection.insert_one(document)
        logger.info(f"Inserted document in {self.collection_name}", extra={"doc_id": str(result.inserted_id)})
        return str(result.inserted_id)
    
    async def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update a single document"""
        result = await self.collection.update_one(filter, {"$set": update})
        return result.modified_count > 0
    
    async def delete_one(self, filter: Dict[str, Any]) -> bool:
        """Delete a single document"""
        result = await self.collection.delete_one(filter)
        return result.deleted_count > 0
    
    async def upsert(self, filter: Dict[str, Any], document: Dict[str, Any]) -> bool:
        """Insert or update a document"""
        result = await self.collection.update_one(filter, {"$set": document}, upsert=True)
        return result.modified_count > 0 or result.upserted_id is not None


class UserRepository(BaseRepository):
    """Repository for user operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "users")
    
    async def create_indexes(self):
        """Create database indexes for performance"""
        await self.collection.create_index("user_id", unique=True)
        await self.collection.create_index("email", unique=True)
        logger.info("Created indexes for users collection")
    
    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user by email"""
        return await self.find_one({"email": email})
    
    async def find_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find user by user_id"""
        user = await self.find_one({"user_id": user_id})
        if not user:
            raise ResourceNotFoundError("User", user_id)
        return user


class SessionRepository(BaseRepository):
    """Repository for session operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "user_sessions")
    
    async def create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index("session_token", unique=True)
        await self.collection.create_index("user_id")
        await self.collection.create_index("expires_at", expireAfterSeconds=0)  # TTL index
        logger.info("Created indexes for user_sessions collection")
    
    async def find_by_token(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Find session by token"""
        return await self.find_one({"session_token": session_token})
    
    async def delete_expired_sessions(self):
        """Clean up expired sessions"""
        result = await self.collection.delete_many({
            "expires_at": {"$lt": datetime.now(timezone.utc)}
        })
        logger.info(f"Deleted {result.deleted_count} expired sessions")
        return result.deleted_count


class ConversationRepository(BaseRepository):
    """Repository for conversation operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "conversations")
    
    async def create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index("conversation_id", unique=True)
        await self.collection.create_index("user_id")
        await self.collection.create_index([("user_id", 1), ("updated_at", -1)])  # Compound index
        logger.info("Created indexes for conversations collection")
    
    async def find_by_user(
        self,
        user_id: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Find conversations by user with pagination"""
        return await self.find_many(
            filter={"user_id": user_id},
            limit=limit,
            skip=skip,
            sort=[("updated_at", -1)]
        )
    
    async def find_by_id_and_user(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find conversation by ID and user (authorization check)"""
        conversation = await self.find_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        if not conversation:
            raise ResourceNotFoundError("Conversation", conversation_id)
        return conversation
    
    async def delete_by_id_and_user(self, conversation_id: str, user_id: str) -> bool:
        """Delete conversation with authorization check"""
        deleted = await self.delete_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        if not deleted:
            raise ResourceNotFoundError("Conversation", conversation_id)
        return deleted
