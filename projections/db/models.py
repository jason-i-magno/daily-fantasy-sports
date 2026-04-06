"""SQLAlchemy ORM models for projections."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ProjectionSource(Base):
    __tablename__ = "projection_sources"

    id = Column(Integer, primary_key=True)
    slate_id = Column(String, nullable=False)
    source = Column(String, nullable=False)
    pulled_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("slate_id", "source", name="uq_projection_source"),)


class PlayerProjection(Base):
    __tablename__ = "player_projections"

    id = Column(Integer, primary_key=True)
    slate_id = Column(String, nullable=False)
    player_name = Column(String, nullable=False)
    team = Column(String, nullable=True)
    position = Column(String, nullable=True)
    projection = Column(Float, nullable=False)
    ceiling = Column(Float, nullable=True)
    floor = Column(Float, nullable=True)
    minutes = Column(Float, nullable=True)
    ownership = Column(Float, nullable=True)
    source = Column(String, nullable=False)
    pulled_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("slate_id", "player_name", "source", name="uq_projection_unique"),
    )
