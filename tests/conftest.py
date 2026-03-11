import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.database import Base, get_db

SQLALCHEMY_TEST_URL = 'sqlite:///./test.db'
engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={'check_same_thread': False})
TestSessionLocal = sessionmaker(bind=engine)

@pytest.fixture(scope='function')
def db_session():
    # Create the tables in the fake database
    Base.metadata.create_all(bind=engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope='function')
def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()