from sqlalchemy import (
    create_engine, event, String, Integer, Float, Text, CheckConstraint,
    UniqueConstraint, ForeignKey, Index, JSON, text, DateTime, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from .base import Base
import numpy as np
from .utils import NumpyArrayJSON




class EffnetEmbeddings(Base):
    __tablename__ = "effnet_embeddings"
    __table_args__ = {"extend_existing": True}
    
    # same sid as metadata
    sid: Mapped[str] = mapped_column(
        String,
        primary_key=True
    )

        
    # 2D array - list of lists of floats
    embedding: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
    