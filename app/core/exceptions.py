"""
Custom application exceptions and centralized exception handlers.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DocumentNotFoundError(AppError):
    def __init__(self, document_id: str):
        super().__init__(f"Document '{document_id}' was not found.", status.HTTP_404_NOT_FOUND)


class InvalidFileError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)


class SessionNotFoundError(AppError):
    def __init__(self, session_id: str):
        super().__init__(f"Session '{session_id}' was not found.", status.HTTP_404_NOT_FOUND)


class EmailAlreadyRegisteredError(AppError):
    def __init__(self):
        super().__init__("An account with this email address already exists.", status.HTTP_400_BAD_REQUEST)


class InvalidCredentialsError(AppError):
    def __init__(self):
        super().__init__("Invalid email or password.", status.HTTP_401_UNAUTHORIZED)


class InvalidTokenError(AppError):
    def __init__(self, message: str = "Invalid or expired authentication token."):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)


class ForbiddenError(AppError):
    def __init__(self, message: str = "You do not have permission to access this resource."):
        super().__init__(message, status.HTTP_403_FORBIDDEN)


class ProcessingError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ConfigurationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)


def _error_response(status_code: int, message: str, details=None) -> JSONResponse:
    body = {"success": False, "error": {"message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach consistent JSON error handling to the FastAPI application."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        logger.warning("AppError on %s: %s", request.url.path, exc.message)
        return _error_response(exc.status_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # exc.errors() can contain non-JSON-serializable objects (e.g. the raw
        # ValueError instance inside "ctx" for custom Pydantic validators), so
        # we sanitize each error entry before including it in the response body.
        safe_errors = []
        for error in exc.errors():
            safe_error = {k: v for k, v in error.items() if k != "ctx"}
            if "ctx" in error and isinstance(error["ctx"], dict):
                safe_error["ctx"] = {k: str(v) for k, v in error["ctx"].items()}
            safe_errors.append(safe_error)

        logger.warning("Validation error on %s: %s", request.url.path, safe_errors)
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "Request validation failed.", safe_errors)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        return _error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url.path)
        return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "An unexpected internal error occurred.")
