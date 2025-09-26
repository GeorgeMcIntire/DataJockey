from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

class Base(DeclarativeBase):
    pass


host = os.getenv('POSTGRES_USER')
port = os.getenv('POSTGRES_PORT')
user_name = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
db_name = os.getenv('POSTGRES_DB')
engine = create_engine(f'postgresql://{user_name}:{password}@{host}:{port}/{db_name}')


def init_db():
    Base.metadata.create_all(engine)