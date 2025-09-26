from sqlalchemy import (
    create_engine, event, String, Integer, Float, Text, CheckConstraint,
    UniqueConstraint, ForeignKey, Index, JSON, text, DateTime, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from .utils import NumpyArrayJSON
import numpy as np
from .base import Base




class Tonal(Base):
    __tablename__ = "tonal"

    # PK
    sid: Mapped[str] = mapped_column(
        String,
        primary_key=True
    )

    # ---- scalars (float) ----
    chords_changes_rate:             Mapped[float] = mapped_column(Float, nullable=False)
    chords_number_rate:              Mapped[float] = mapped_column(Float, nullable=False)
    chords_strength_mean:            Mapped[float] = mapped_column(Float, nullable=False)
    hpcp_crest_mean:                 Mapped[float] = mapped_column(Float, nullable=False)
    hpcp_entropy_mean:               Mapped[float] = mapped_column(Float, nullable=False)
    key_edma_strength:               Mapped[float] = mapped_column(Float, nullable=False)
    key_krumhansl_strength:          Mapped[float] = mapped_column(Float, nullable=False)
    key_temperley_strength:          Mapped[float] = mapped_column(Float, nullable=False)
    tuning_diatonic_strength:        Mapped[float] = mapped_column(Float, nullable=False)
    tuning_equal_tempered_deviation: Mapped[float] = mapped_column(Float, nullable=False)
    tuning_frequency:                Mapped[float] = mapped_column(Float, nullable=False)
    tuning_nontempered_energy_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    # ---- vectors (Postgres numeric arrays) ----
    hpcp_mean:            Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    chords_histogram:     Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    thpcp:                Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)

    # ---- categorical strings ----
    chords_key:           Mapped[str] = mapped_column(String(8),  nullable=False)  # e.g., 'C#m'
    chords_scale:         Mapped[str] = mapped_column(String(8),  nullable=False)  # 'major'/'minor'
    key_edma_key:         Mapped[str] = mapped_column(String(8),  nullable=False)
    key_edma_scale:       Mapped[str] = mapped_column(String(8),  nullable=False)
    key_krumhansl_key:    Mapped[str] = mapped_column(String(8),  nullable=False)
    key_krumhansl_scale:  Mapped[str] = mapped_column(String(8),  nullable=False)
    key_temperley_key:    Mapped[str] = mapped_column(String(8),  nullable=False)
    key_temperley_scale:  Mapped[str] = mapped_column(String(8),  nullable=False)