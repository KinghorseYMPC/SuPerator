import sys
from pathlib import Path
import re
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path(request):
    """Workspace-local tmp_path replacement for restricted Windows temp dirs."""

    base = ROOT / "outputs" / "test_tmp"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    safe_name = f"{safe_name[:32]}_{uuid.uuid4().hex[:12]}"
    target = (base / safe_name).resolve()
    base_resolved = base.resolve()
    if base_resolved not in target.parents:
        raise RuntimeError(f"Refusing to create temp path outside {base_resolved}: {target}")
    target.mkdir(parents=True, exist_ok=True)
    yield target
