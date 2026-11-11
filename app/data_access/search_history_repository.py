"""SearchHistoryRepository (entity access): reads and writes search history."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.control.text_utils import normalize_keyword
from app.domain.models import ImageSearchQuery, SearchHistory, VoiceSearchQuery


class SearchHistoryRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_recent_searches(self, customer_id: int, limit: int = 5) -> list[str]:
        """getRecentSearches(customerId): distinct texts (ignoring case, accents and
        spaces), most recent first, each shown as it was last typed."""
        stmt = (
            select(SearchHistory.query_text)
            .where(SearchHistory.customer_id == customer_id,
                   SearchHistory.search_type.in_(["KEYWORD", "VOICE"]),
                   SearchHistory.query_text.is_not(None))
            .order_by(SearchHistory.search_id.desc())
            .limit(500)
        )
        recent: dict[str, str] = {}
        for text in self.session.scalars(stmt):
            recent.setdefault(normalize_keyword(text), text)
            if len(recent) == limit:
                break
        return list(recent.values())

    def save(self, history: SearchHistory,
             detail: VoiceSearchQuery | ImageSearchQuery | None = None) -> SearchHistory:
        """saveSearchHistory(...): one history row plus its optional voice/image detail."""
        self.session.add(history)
        self.session.flush()
        if detail is not None:
            detail.search_id = history.search_id
            self.session.add(detail)
        self.session.commit()
        return history
