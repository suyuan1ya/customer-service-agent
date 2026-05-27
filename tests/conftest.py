import pytest
from fastapi.testclient import TestClient
from tools.database import init_db


@pytest.fixture(autouse=True)
def setup_db():
    """每个测试前确保数据库已初始化"""
    init_db()


@pytest.fixture
def client():
    from main import app
    return TestClient(app)
