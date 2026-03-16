"""OMEGA database engine and session utilities."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine(url: str, **kwargs):
    """Create a SQLAlchemy engine for the given database URL."""
    return create_engine(url, **kwargs)


def get_session_factory(engine):
    """Return a sessionmaker bound to *engine*."""
    return sessionmaker(bind=engine)
