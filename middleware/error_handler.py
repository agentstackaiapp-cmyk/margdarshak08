"""
Global Error Handler Middleware
FAANG-Level: Never expose technical errors to users
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from typing import Union

logger = logging.getLogger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler that NEVER exposes technical details to users
    """
    
    # Log the full error for debugging (server-side only)
    logger.error(
        f"Error processing request: {request.method} {request.url.path}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__
        }
    )
    
    # Return user-friendly messages (NEVER show code/stack traces)
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.detail,
                "error_code": "API_ERROR"
            }
        )
    
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": "Invalid request data. Please check your input.",
                "error_code": "VALIDATION_ERROR"
            }
        )
    
    # For all other errors: Generic user-friendly message
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "We're experiencing technical difficulties. Please try again in a moment.",
            "error_code": "SERVER_ERROR"
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions with user-friendly messages"""
    
    # Map status codes to user-friendly messages
    user_messages = {
        401: "Please sign in to continue",
        403: "You don't have permission to access this",
        404: "The requested item was not found",
        429: "Too many requests. Please wait a moment and try again",
        500: "Something went wrong on our end. Please try again",
        503: "Service temporarily unavailable. Please try again shortly"
    }
    
    message = user_messages.get(exc.status_code, exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": message,
            "error_code": "HTTP_ERROR"
        }
    )
