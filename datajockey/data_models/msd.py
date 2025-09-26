from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base
import numpy as np
from .utils import NumpyArrayJSON


class MSDEmbeddings(Base):
	__tablename__ = "msd_embeddings"
	__table_args__ = {"extend_existing": True}
	
	# same sid as metadata
	sid: Mapped[str] = mapped_column(
		String,
		primary_key=True
	)
	
	
	# 2D array - list of lists of floats
	embedding: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	
	
class MSDEmotions(Base):
	__tablename__ = "msd_emotions"
	__table_args__ = {"extend_existing": True}
	
	# same sid as metadata
	sid: Mapped[str] = mapped_column(
		String,
		primary_key=True
	)
	
	
	# 2D array - list of lists of floats
	valence: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	arousal: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False)
	
	
class MSDMoods(Base):
	__tablename__ = "msd_moods"
	__table_args__ = {"extend_existing": True}
	
	# same sid as metadata
	sid: Mapped[str] = mapped_column(
		String,
		primary_key=True
	)
	
	
	
	rousing: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False, info= {"description":
																				"passionate, rousing, confident, boisterous, and rowdy"})
		
		
	cheerful: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False, 
													info= {"description":"rollicking, fun, sweet, amiable, good natured"})
	
	wistful: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False,
												info= {"description":"literate, poignant, autumnal, brooding"})
	
	silly: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False,
												info= {"description":"funny, campy, quirk, whimsical, witty, wry"})
	
	
	intense: Mapped[np.ndarray] = mapped_column(NumpyArrayJSON, nullable=False,
												info= {"description":"aggressive,fiery, tense, anxious, visceral, volatile"})
	

attrs = {
    "__tablename__": "jamendo_moodtheme",
    "__table_args__": {"extend_existing": True},  # notebook-friendly
    "sid": mapped_column(String, primary_key=True),
}

jamendo_cols = ['action', 'adventure', 'advertising', 'background', 'ballad', 'calm', 'children', 'christmas', 'commercial', 'cool', 'corporate', 'dark', 'deep', 'documentary', 'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'epic', 'fast', 'film', 'fun', 'funny', 'game', 'groovy', 'happy', 'heavy', 'holiday', 'hopeful', 'inspiring', 'love', 'meditative', 'melancholic', 'melodic', 'motivational', 'movie', 'nature', 'party', 'positive', 'powerful', 'relaxing', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape', 'space', 'sport', 'summer', 'trailer', 'travel', 'upbeat', 'uplifting']

for col in jamendo_cols:
    attrs[col] = mapped_column(NumpyArrayJSON, nullable=False)

JamendoMoodTheme = type("JamendoMoodTheme", (Base,), attrs)