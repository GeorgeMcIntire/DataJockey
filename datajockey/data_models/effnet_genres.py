from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base
import numpy as np
from .utils import NumpyArrayJSON

# list of genre columns
EFFNET_GENRE_COLS = ['electronic___abstract', 'electronic___acid',
       'electronic___acid_house', 'electronic___bassline',
       'electronic___berlin_school', 'electronic___big_beat',
       'electronic___breakbeat', 'electronic___breakcore',
       'electronic___breaks', 'electronic___dance_pop',
       'electronic___deep_house', 'electronic___deep_techno',
       'electronic___disco', 'electronic___disco_polo',
       'electronic___downtempo', 'electronic___dub',
       'electronic___dub_techno', 'electronic___ebm',
       'electronic___electro', 'electronic___electro_house',
       'electronic___electroclash', 'electronic___euro_house',
       'electronic___euro_disco', 'electronic___eurobeat',
       'electronic___eurodance', 'electronic___experimental',
       'electronic___garage_house', 'electronic___ghetto',
       'electronic___ghetto_house', 'electronic___halftime',
       'electronic___hands_up', 'electronic___hard_house',
       'electronic___hard_techno', 'electronic___hardcore',
       'electronic___hardstyle', 'electronic___hi_nrg',
       'electronic___hip_hop', 'electronic___hip_house',
       'electronic___house', 'electronic___idm',
       'electronic___industrial', 'electronic___italo_house',
       'electronic___italo_disco', 'electronic___italodance',
       'electronic___juke', 'electronic___jumpstyle',
       'electronic___latin', 'electronic___minimal',
       'electronic___minimal_techno', 'electronic___new_age',
       'electronic___new_beat', 'electronic___new_wave',
       'electronic___nu_disco', 'electronic___progressive_house',
       'electronic___rhythmic_noise', 'electronic___synth_pop',
       'electronic___synthwave', 'electronic___tech_house',
       'electronic___techno', 'electronic___tribal',
       'electronic___tribal_house', 'electronic___tropical_house',
       'electronic___uk_garage', 'electronic___vaporwave',
       'folk_world__country___african',
       'folk_world__country___canzone_napoletana',
       'folk_world__country___catalan_music',
       'folk_world__country___flamenco', 'folk_world__country___folk',
       'folk_world__country___highlife',
       'folk_world__country___hindustani',
       'folk_world__country___honky_tonk',
       'folk_world__country___soukous', 'folk_world__country___séga',
       'folk_world__country___zouk', 'funk__soul___afrobeat',
       'funk__soul___boogie', 'funk__soul___contemporary_rb',
       'funk__soul___disco', 'funk__soul___free_funk',
       'funk__soul___funk', 'funk__soul___neo_soul',
       'funk__soul___new_jack_swing', 'funk__soul___p.funk',
       'funk__soul___psychedelic', 'funk__soul___rhythm__blues',
       'funk__soul___soul', 'funk__soul___uk_street_soul',
       'jazz___afrobeat', 'jazz___bossa_nova', 'latin___afro_cuban',
       'latin___batucada', 'latin___beguine', 'latin___bolero',
       'latin___boogaloo', 'latin___bossanova', 'latin___cha_cha',
       'latin___compas', 'latin___cubano', 'latin___cumbia',
       'latin___forró', 'latin___guajira', 'latin___guaracha',
       'latin___mpb', 'latin___mambo', 'latin___pachanga',
       'latin___porro', 'latin___ranchera', 'latin___reggaeton',
       'latin___rumba', 'latin___salsa', 'latin___samba', 'latin___son',
       'pop___bubblegum', 'pop___city_pop', 'pop___europop',
       'pop___indie_pop', 'pop___j_pop', 'reggae___calypso',
       'reggae___dancehall', 'reggae___dub', 'reggae___lovers_rock',
       'reggae___ragga', 'reggae___reggae', 'reggae___reggae_pop',
       'reggae___rocksteady', 'reggae___roots_reggae', 'reggae___soca',
       'latin___baião']

# build attributes dynamically
attrs = {
    "__tablename__": "effnet_genres",
    "__table_args__": {"extend_existing": True},  # notebook-friendly
    "sid": mapped_column(String, primary_key=True),
}

for col in EFFNET_GENRE_COLS:
    attrs[col] = mapped_column(Float, nullable=False)

EffnetGenres = type("EffnetGenres", (Base,), attrs)
