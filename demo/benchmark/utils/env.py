from __future__ import annotations

from pathlib import Path
from typing import List


def load_env() -> List[Path]:
    """
    Load environment variables from a .env file if python-dotenv is installed.

    We support two common run modes:
    - run inside `demo/` (cwd=demo) while `.env` sits at repo root
    - run from repo root (cwd=repo) with `.env` at repo root
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return []

    candidates: List[Path] = []

    # 1) Current working directory
    candidates.append(Path.cwd() / ".env")

    # 2) Walk up from this file location a few levels to find repo root `.env`
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        candidates.append(p / ".env")
        if len(candidates) >= 10:
            break

    loaded: List[Path] = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)
            loaded.append(path)
    return loaded

