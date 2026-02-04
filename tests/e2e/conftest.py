import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def force_mock_mode():
    """
    Force le mode mock pour tous les tests E2E.
    Garantit zéro appel API réel.
    """
    os.environ["LLM_MODE"] = "mock"
    yield
    os.environ.pop("LLM_MODE", None)
