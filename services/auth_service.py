"""
Authentication Service
FAANG-Level: Business logic separation, proper error handling
"""
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import uuid
import httpx
from core.config import settings
from core.exceptions import AuthenticationError, ExternalServiceError
from core.logging import get_logger
from repositories import UserRepository, SessionRepository

logger = get_logger(__name__)


class AuthService:
    """Service for authentication operations"""
    
    def __init__(self, user_repo: UserRepository, session_repo: SessionRepository):
        self.user_repo = user_repo
        self.session_repo = session_repo
    
    async def exchange_session_id(self, session_id: str) -> Dict[str, Any]:
        """
        Exchange Emergent OAuth session_id for user data
        
        Args:
            session_id: Session ID from Emergent OAuth
            
        Returns:
            User data dictionary
            
        Raises:
            AuthenticationError: If session_id is invalid
            ExternalServiceError: If Emergent service is unavailable
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    settings.EMERGENT_AUTH_URL,
                    headers={"X-Session-ID": session_id}
                )
                
                if response.status_code != 200:
                    logger.warning(f"Invalid session_id attempt: {session_id[:10]}...")
                    raise AuthenticationError("Invalid session ID")
                
                user_data = response.json()
                logger.info(f"Successfully exchanged session_id for user: {user_data.get('email')}")
                return user_data
                
        except httpx.RequestError as e:
            logger.error(f"Emergent auth service error: {str(e)}")
            raise ExternalServiceError("Emergent Auth", str(e))
    
    async def create_or_update_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create new user or update existing user
        
        Args:
            user_data: User data from OAuth provider
            
        Returns:
            user_id
        """
        email = user_data["email"]
        existing_user = await self.user_repo.find_by_email(email)
        
        if existing_user:
            # Update user info
            user_id = existing_user["user_id"]
            await self.user_repo.update_one(
                {"user_id": user_id},
                {
                    "name": user_data.get("name", existing_user.get("name")),
                    "picture": user_data.get("picture", existing_user.get("picture"))
                }
            )
            logger.info(f"Updated existing user: {email}")
        else:
            # Create new user
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            new_user = {
                "user_id": user_id,
                "email": email,
                "name": user_data.get("name", ""),
                "picture": user_data.get("picture"),
                "created_at": datetime.now(timezone.utc)
            }
            await self.user_repo.insert_one(new_user)
            logger.info(f"Created new user: {email}")
        
        return user_id
    
    async def create_session(self, user_id: str, session_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user session
        
        Args:
            user_id: User ID
            session_token: Optional custom session token
            
        Returns:
            Session data dictionary
        """
        if not session_token:
            session_token = f"session_{uuid.uuid4().hex}"
        
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.SESSION_EXPIRE_DAYS)
        
        session = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at,
            "created_at": datetime.now(timezone.utc)
        }
        
        await self.session_repo.insert_one(session)
        logger.info(f"Created session for user: {user_id}")
        
        return session
    
    async def validate_session(self, session_token: str) -> Dict[str, Any]:
        """
        Validate session token and return user
        
        Args:
            session_token: Session token from cookie/header
            
        Returns:
            User data dictionary
            
        Raises:
            AuthenticationError: If session is invalid or expired
        """
        session = await self.session_repo.find_by_token(session_token)
        
        if not session:
            raise AuthenticationError("Invalid session")
        
        # Check expiry
        expires_at = session["expires_at"]
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        
        if expires_at < datetime.now(timezone.utc):
            raise AuthenticationError("Session expired")
        
        # Get user
        user = await self.user_repo.find_by_user_id(session["user_id"])
        return user
    
    async def delete_session(self, session_token: str) -> bool:
        """Delete a session (logout)"""
        deleted = await self.session_repo.delete_one({"session_token": session_token})
        if deleted:
            logger.info(f"Deleted session: {session_token[:10]}...")
        return deleted
