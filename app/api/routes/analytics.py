from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.services.analytics_service import AnalyticsService
from app.schemas.analytics import AnalyticsResponse
from app.core.cache import get_cache_manager, make_cache_key

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# A short, separate TTL-namespace cache is used for analytics: cheap to
# recompute but frequently requested, and using a short fixed TTL here
# (rather than the general document-mutation invalidation used for
# summaries/comparisons) is an acceptable "eventually fresh" tradeoff for a
# dashboard-style endpoint.
_ANALYTICS_NAMESPACE = "analytics"


@router.get("", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Returns usage statistics scoped to the current user's own documents and queries.
    Cached briefly per-user (see CACHE_TTL_SECONDS) since analytics are read far more
    often than they change."""
    cache = get_cache_manager()
    cache_key = make_cache_key(_ANALYTICS_NAMESPACE, current_user.id)
    cached = cache.get(cache_key)
    if cached is not None:
        return AnalyticsResponse(**cached)

    service = AnalyticsService(db)
    data = service.get_analytics(user_id=current_user.id)
    response = AnalyticsResponse(**data)
    cache.set(cache_key, response.model_dump())
    return response
