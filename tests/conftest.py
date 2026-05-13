import sys
from pathlib import Path
from unittest.mock import MagicMock

SYSTEM_DIR = Path(__file__).resolve().parent.parent / "multi-agent_system"
sys.path.insert(0, str(SYSTEM_DIR))

for mod_name in [
    "langchain_core",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_community",
    "langchain_community.vectorstores",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import agents.cardiologist  # noqa: E402, F401
