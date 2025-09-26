from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

attrs = {
    "__tablename__": "mule",
    "__table_args__": {"extend_existing": True},  # notebook-friendly
    "sid": mapped_column(String, primary_key=True),
}

for i in range(1, 1729):
    attrs[f"embed_{i}"] = mapped_column(Float, nullable=False)


MuleEmbeddings = type("MuleEmbeddings", (Base,), attrs)