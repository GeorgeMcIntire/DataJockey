from sqlalchemy import (
    create_engine, event, String, Integer, Float, Text, CheckConstraint,
    UniqueConstraint, ForeignKey, Index, JSON, text, DateTime, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from .utils import NumpyArrayJSON
import numpy as np
from .base import Base




class Rhythm(Base):

    __tablename__ = "rhythm"
    __table_args__ = (
        CheckConstraint("beats_count >= 0", name="ck_beats_count_nonneg"),
        CheckConstraint("danceability >= 0 AND danceability <= 1", name="ck_danceability_0_1"),
        CheckConstraint("onset_rate >= 0", name="ck_onset_rate_nonneg"),
        CheckConstraint("bpm_histogram_first_peak_weight >= 0 AND bpm_histogram_first_peak_weight <= 1",
                        name="ck_bpm1_weight_0_1"),
        CheckConstraint("bpm_histogram_second_peak_weight >= 0 AND bpm_histogram_second_peak_weight <= 1",
                        name="ck_bpm2_weight_0_1"),
        CheckConstraint("bpm_histogram_second_peak_spread >= 0", name="ck_bpm2_spread_nonneg"),
        {"extend_existing": True},  # notebook-friendly
    )

    # PK & FK to metadata
    sid: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    # -------- scalars --------
    beats_count:                          Mapped[float] = mapped_column(Float, nullable=False)
    beats_loudness_mean:                  Mapped[float] = mapped_column(Float, nullable=False)   # may be negative dB
    bpm:                                  Mapped[float] = mapped_column(Float, nullable=False)
    bpm_histogram_first_peak_bpm:         Mapped[float] = mapped_column(Float, nullable=False)
    bpm_histogram_first_peak_weight:      Mapped[float] = mapped_column(Float, nullable=False)
    bpm_histogram_second_peak_bpm:        Mapped[float] = mapped_column(Float, nullable=False)
    bpm_histogram_second_peak_spread:     Mapped[float] = mapped_column(Float, nullable=False)
    bpm_histogram_second_peak_weight:     Mapped[float] = mapped_column(Float, nullable=False)
    danceability:                         Mapped[float] = mapped_column(Float, nullable=False)
    onset_rate:                           Mapped[float] = mapped_column(Float, nullable=False)

    # -------- vectors (JSONB) --------
    beats_loudness_band_ratio_mean:       Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    beats_position:                       Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    bpm_histogram:                        Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)