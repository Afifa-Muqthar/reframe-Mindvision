"""
Local-development-only SSL verification configuration for Hugging Face downloads.

Opt-in workaround for environments (e.g. Windows machines with corporate proxies,
antivirus SSL inspection, or missing system root certs in Python) where Python requests
fails SSL certificate verification against huggingface.co.

This is strictly disabled by default. It only activates when the environment variable
`HF_DISABLE_SSL_VERIFY=true` is explicitly set in the environment or `.env`.
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_configured = False


def configure_hf_ssl_verify() -> bool:
    """
    Configures Hugging Face HTTP backend to bypass SSL verification if opt-in enabled.
    Returns True if workaround was activated, False otherwise.
    """
    global _configured
    if _configured:
        return True

    # Load .env if present
    load_dotenv()

    disable_ssl = os.getenv("HF_DISABLE_SSL_VERIFY", "").strip().lower() in ("true", "1", "yes")
    if not disable_ssl:
        return False

    try:
        import urllib3
        import requests
        from huggingface_hub import configure_http_backend
        from huggingface_hub.utils._http import _default_backend_factory

        # Suppress noisy InsecureRequestWarning when opt-in is active
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        def insecure_backend_factory() -> requests.Session:
            session = _default_backend_factory()
            session.verify = False
            return session

        configure_http_backend(backend_factory=insecure_backend_factory)
        logger.warning(
            "[DEV NOTICE] HF_DISABLE_SSL_VERIFY is enabled. Hugging Face downloads are running with verify=False. "
            "Do NOT use this in production."
        )
        _configured = True
        return True
    except Exception as e:
        logger.warning(f"Could not configure Hugging Face HTTP backend: {e}")
        return False
