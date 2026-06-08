import pytest


@pytest.fixture(scope="session", autouse=True)
def flush_langfuse():
    yield
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
