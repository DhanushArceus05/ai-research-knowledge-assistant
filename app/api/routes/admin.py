from fastapi import APIRouter, Depends

from app.api.dependencies import get_db, require_admin
from app.database.models import User
from app.services.analytics_service import AnalyticsService
from app.schemas.analytics import AnalyticsResponse
from app.core.cache import get_cache_manager

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/analytics", response_model=AnalyticsResponse)
def get_system_analytics(db=Depends(get_db), current_user: User = Depends(require_admin)):
    """System-wide analytics across every user's documents. Requires the ADMIN role."""
    service = AnalyticsService(db)
    data = service.get_analytics(user_id=None)
    return AnalyticsResponse(**data)


@router.get("/cache")
def get_cache_diagnostics(current_user: User = Depends(require_admin)):
    """Safe, non-sensitive cache diagnostics (backend mode, TTL, size). Requires the ADMIN role."""
    cache = get_cache_manager()
    return {"success": True, "cache": cache.diagnostics()}
