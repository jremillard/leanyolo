import os
import sys


def _add_repo_root_to_sys_path():
    # Ensure imports like `import leanyolo` resolve when running tests directly.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_add_repo_root_to_sys_path()
