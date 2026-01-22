from __future__ import annotations

import os
from pathlib import Path
from typing import List


def load_env() -> List[Path]:
    """
    Load environment variables from a .env file if python-dotenv is installed.

    We support two common run modes:
    - run inside `demo/` (cwd=demo) while `.env` sits at repo root
    - run from repo root (cwd=repo) with `.env` at repo root
    """
    def _manual_load_dotenv(path: Path) -> None:
        """
        Minimal .env loader (fallback when python-dotenv isn't installed).
        - Supports KEY=VALUE lines
        - Ignores blank lines and comments starting with '#'
        - Strips surrounding single/double quotes
        - Does not override existing env vars
        """
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v

    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore
    except Exception:
        _load_dotenv = None

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
            if _load_dotenv is not None:
                _load_dotenv(dotenv_path=path, override=False)
            else:
                _manual_load_dotenv(path)
            loaded.append(path)
    return loaded
