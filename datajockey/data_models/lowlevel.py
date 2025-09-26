from sqlalchemy import (
    create_engine, event, String, Integer, Float, Text, CheckConstraint,
    UniqueConstraint, ForeignKey, Index, JSON, text, DateTime, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
import numpy as np
from .base import Base
from .utils import NumpyArrayJSON


class LowLevel(Base):
    __tablename__ = "lowlevel"
    __table_args__ = {"extend_existing": True}
    
    # same sid as metadata
    sid: Mapped[str] = mapped_column(
        String,
        primary_key=True
    )
    
    # ---- scalar features ----
    average_loudness: Mapped[float] = mapped_column(Float, nullable=False)
    barkbands_crest_mean: Mapped[float] = mapped_column(Float, nullable=False)
    barkbands_flatness_db_mean: Mapped[float] = mapped_column(Float, nullable=False)
    barkbands_kurtosis_mean: Mapped[float] = mapped_column(Float, nullable=False)
    barkbands_skewness_mean: Mapped[float] = mapped_column(Float, nullable=False)
    barkbands_spread_mean: Mapped[float] = mapped_column(Float, nullable=False)
    dissonance_mean: Mapped[float] = mapped_column(Float, nullable=False)
    dynamic_complexity: Mapped[float] = mapped_column(Float, nullable=False)
    erbbands_crest_mean: Mapped[float] = mapped_column(Float, nullable=False)
    erbbands_flatness_db_mean: Mapped[float] = mapped_column(Float, nullable=False)
    erbbands_kurtosis_mean: Mapped[float] = mapped_column(Float, nullable=False)
    erbbands_skewness_mean: Mapped[float] = mapped_column(Float, nullable=False)
    erbbands_spread_mean: Mapped[float] = mapped_column(Float, nullable=False)
    hfc_mean: Mapped[float] = mapped_column(Float, nullable=False)
    loudness_ebu128_integrated: Mapped[float] = mapped_column(Float, nullable=False)
    loudness_ebu128_loudness_range: Mapped[float] = mapped_column(Float, nullable=False)
    loudness_ebu128_momentary_mean: Mapped[float] = mapped_column(Float, nullable=False)
    loudness_ebu128_short_term_mean: Mapped[float] = mapped_column(Float, nullable=False)
    melbands_crest_mean: Mapped[float] = mapped_column(Float, nullable=False)
    melbands_flatness_db_mean: Mapped[float] = mapped_column(Float, nullable=False)
    melbands_kurtosis_mean: Mapped[float] = mapped_column(Float, nullable=False)
    melbands_skewness_mean: Mapped[float] = mapped_column(Float, nullable=False)
    melbands_spread_mean: Mapped[float] = mapped_column(Float, nullable=False)
    pitch_salience_mean: Mapped[float] = mapped_column(Float, nullable=False)
    silence_rate_20dB_mean: Mapped[float] = mapped_column(Float, nullable=False)
    silence_rate_30dB_mean: Mapped[float] = mapped_column(Float, nullable=False)
    silence_rate_60dB_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_centroid_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_complexity_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_decrease_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_energy_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_energyband_high_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_energyband_low_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_energyband_middle_high_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_energyband_middle_low_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_entropy_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_flux_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_kurtosis_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_rms_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_rolloff_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_skewness_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_spread_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spectral_strongpeak_mean: Mapped[float] = mapped_column(Float, nullable=False)
    zerocrossingrate_mean: Mapped[float] = mapped_column(Float, nullable=False)
    
    # ---- array features stored as JSON ----
    barkbands_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    erbbands_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    gfcc_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    melbands_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    melbands128_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    mfcc_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    spectral_contrast_coeffs_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    spectral_contrast_valleys_mean: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)