import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from jvl.db.models import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def minimal_config_data() -> dict:
    return {
        "default_backend": "ollama",
        "backends": {
            "ollama": {"model": "phi3:3.8b", "base_url": "http://localhost:11434"},
        },
    }


@pytest.fixture
def full_config_data() -> dict:
    return {
        "default_backend": "ollama",
        "backends": {
            "ollama": {"model": "phi3:3.8b", "base_url": "http://localhost:11434"},
            "openai": {"model": "gpt-4o", "api_key": "sk-test"},
            "anthropic": {"model": "claude-sonnet-4-5", "api_key": "ant-test"},
            "azure": {
                "model": "gpt-4o",
                "api_key": "az-test",
                "api_base": "https://test.openai.azure.com",
                "api_version": "2024-02-01",
            },
            "gemini": {"model": "gemini-2.5-flash", "api_key": "gem-test"},
        },
    }
