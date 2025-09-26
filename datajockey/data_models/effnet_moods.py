from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base
from sqlalchemy.dialects.postgresql import JSONB
from .utils import NumpyArrayJSON
import numpy as np

class EffnetMoods(Base):
	__tablename__ = "effnet_moods"
	__table_args__ = {"extend_existing": True}
	
	# same sid as metadata
	sid: Mapped[str] = mapped_column(
		String,
		primary_key=True
	)        
	approachable: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	danceable: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	engagement: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	acoustic: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	aggressive: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	happy: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	party: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	sad: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	timbre_bright: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)