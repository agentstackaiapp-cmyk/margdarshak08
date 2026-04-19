"""
Custom Exception Classes
FAANG-Level: Proper exception hierarchy with context
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseAppException(HTTPException):
    """Base exception for all application exceptions"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or self.__class__.__name__


class AuthenticationError(BaseAppException):
    """Raised when authentication fails"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTH_ERROR"
        )


class AuthorizationError(BaseAppException):
    """Raised when user lacks permissions"""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHZ_ERROR"
        )


class ResourceNotFoundError(BaseAppException):
    """Raised when requested resource doesn't exist"""
    
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id '{resource_id}' not found",
            error_code="NOT_FOUND"
        )


class ValidationError(BaseAppException):
    """Raised when input validation fails"""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )


class ExternalServiceError(BaseAppException):
    """Raised when external service call fails"""
    
    def __init__(self, service: str, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service} service error: {detail}",
            error_code="EXTERNAL_SERVICE_ERROR"
        )


class RateLimitError(BaseAppException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_ERROR",
            headers={"Retry-After": "60"}
        )
