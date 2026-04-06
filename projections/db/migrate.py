"""Create database tables."""

from __future__ import annotations

from projections.db.models import Base
from projections.db.session import get_engine
from projections.config import DEFAULT_DB_URL


def run(db_url: str = DEFAULT_DB_URL) -> None:
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    run()
