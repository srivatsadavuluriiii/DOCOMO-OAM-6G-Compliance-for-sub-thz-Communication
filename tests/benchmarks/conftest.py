import pytest


@pytest.fixture
def benchmark():
    """
    Provide a no-op `benchmark` fixture if pytest-benchmark is not installed.
    It simply calls the function once.
    """
    def _bench(func, *args, **kwargs):
        return func(*args, **kwargs)
    return _bench


