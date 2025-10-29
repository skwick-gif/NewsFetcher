import os


def get_backend_base_url() -> str:
    """Return the FastAPI backend base URL from env or default.

    Reads FASTAPI_BACKEND to allow overrides; defaults to localhost:8000.
    """
    return os.getenv('FASTAPI_BACKEND', 'http://localhost:8000')
