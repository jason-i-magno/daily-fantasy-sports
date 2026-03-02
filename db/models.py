from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "player"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(255), unique=True)


class Contest(Base):
    __tablename__ = "contest"
    __table_args__ = (UniqueConstraint("date", "slate_type"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    slate_type: Mapped[str] = mapped_column(String(255))
    slate_games: Mapped[int] = mapped_column(Integer())
    date: Mapped[datetime] = mapped_column(DateTime())
    has_late_swaps: Mapped[bool] = mapped_column(Boolean())
    missing_late_swap_projections: Mapped[bool] = mapped_column(Boolean())


class Salary(Base):
    __tablename__ = "salary"
    contest_id: Mapped[int] = mapped_column(ForeignKey("contest.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"), primary_key=True)
    salary: Mapped[int] = mapped_column(Integer())
    position: Mapped[str] = mapped_column(String(255))
    team: Mapped[str] = mapped_column(String(3))
    game: Mapped[str] = mapped_column(String(255))


class Projection(Base):
    __tablename__ = "projection"
    contest_id: Mapped[int] = mapped_column(ForeignKey("contest.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"), primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), primary_key=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(), primary_key=True)
    proj_minutes: Mapped[float] = mapped_column(Float())
    proj_fpts: Mapped[float] = mapped_column(Float())


class Source(Base):
    __tablename__ = "source"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class History(Base):
    __tablename__ = "history"
    contest_id: Mapped[int] = mapped_column(ForeignKey("contest.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"), primary_key=True)
    fpts: Mapped[float] = mapped_column(Float())


class Lineup(Base):
    __tablename__ = "lineup"
    __table_args__ = (
        UniqueConstraint(
            "contest_id",
            "strategy",
            "snapshot_time",
        ),
    )

    lineup_id: Mapped[int] = mapped_column(primary_key=True)
    contest_id: Mapped[int] = mapped_column(ForeignKey("contest.id"))
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"))
    snapshot_time: Mapped[datetime] = mapped_column(DateTime())
    is_late_swap: Mapped[bool] = mapped_column(Boolean())
    strategy: Mapped[str] = mapped_column(String(255))


class PlayerLineup(Base):
    __tablename__ = "player_lineup"
    lineup_id: Mapped[int] = mapped_column(
        ForeignKey("lineup.lineup_id"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(Integer(), primary_key=True)
    pg: Mapped[int] = mapped_column(ForeignKey("player.id"))
    sg: Mapped[int] = mapped_column(ForeignKey("player.id"))
    sf: Mapped[int] = mapped_column(ForeignKey("player.id"))
    pf: Mapped[int] = mapped_column(ForeignKey("player.id"))
    c: Mapped[int] = mapped_column(ForeignKey("player.id"))
    g: Mapped[int] = mapped_column(ForeignKey("player.id"))
    f: Mapped[int] = mapped_column(ForeignKey("player.id"))
    util: Mapped[int] = mapped_column(ForeignKey("player.id"))
