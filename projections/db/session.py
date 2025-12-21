"""Session/engine helpers."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from projections.config import DEFAULT_DB_URL


def get_engine(db_url: str = DEFAULT_DB_URL):
    return create_engine(db_url, future=True)


def get_session_maker(db_url: str = DEFAULT_DB_URL):
    engine = get_engine(db_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
