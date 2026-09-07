from __future__ import annotations

import os

os.environ["CAREEROPS_DATABASE_URL"] = "sqlite:///./test_careerops.db"
os.environ["CAREEROPS_CHECKPOINT_DB"] = "test_careerops_checkpoints.db"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
