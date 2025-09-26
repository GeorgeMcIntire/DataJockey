from sqlalchemy import (
    create_engine, event, String, Integer, Float, Text, CheckConstraint,
    UniqueConstraint, ForeignKey, Index, JSON, text, DateTime, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

class Base(DeclarativeBase):
    pass


class Metadata(Base):
    __tablename__ = "metadata"

    __table_args__ = (
        CheckConstraint("length > 1.0 AND length <= 36000.0", name="ck_length"),
        CheckConstraint("bpm >=0 AND bpm < 250.0", name="ck_bpm"),
        UniqueConstraint("rekordbox_id", name="uq_rekordbox_id"),
        UniqueConstraint("sid", name="uq_song_id"),
        UniqueConstraint("path", name="uq_path"),
    )

    # sid from CSV = primary key
    sid: Mapped[str] = mapped_column(String, primary_key=True)

    path: Mapped[str]         = mapped_column(Text, nullable=False)
    rekordbox_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str]        = mapped_column(Text, nullable=False)
    artist: Mapped[str | None]= mapped_column(Text)
    album: Mapped[str | None] = mapped_column(Text)
    genre: Mapped[str | None] = mapped_column(Text)
    dateadded: Mapped[datetime] = mapped_column(DateTime())
    length: Mapped[float]     = mapped_column(Float, nullable=False)       # seconds
    bpm: Mapped[float]        = mapped_column(Float, nullable=False)
    initialkey: Mapped[str]   = mapped_column(String(32), nullable=False)
    
