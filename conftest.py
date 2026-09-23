import sys
import importlib.util
from pathlib import Path

# Ensure root directory is on sys.path for test discovery
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_course_module(name: str, rel_path: str):
    """Cleanly loads a module from a hyphenated directory without deprecation warnings."""
    file_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

