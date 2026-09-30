"""Test-wide environment, applied before any `app` module is imported."""
from __future__ import annotations

import os
import tempfile

# Keep test runs out of the real event log, and never ship test traces to
# LangSmith even when the developer's .env turns tracing on (load_dotenv does
# not override variables that are already set).
os.environ["MEDIBOT_LOG_DIR"] = tempfile.mkdtemp(prefix="medibot-test-logs-")
os.environ["LANGSMITH_TRACING"] = "false"
