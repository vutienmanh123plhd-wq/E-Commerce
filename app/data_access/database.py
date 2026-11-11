"""Engine and session helpers for the SQLite database."""
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.models import Base


def make_engine(url: str) -> Engine:
    return create_engine(url, connect_args={"check_same_thread": False})


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
