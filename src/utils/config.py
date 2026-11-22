"""Config loading and project paths."""
from functools import lru_cache
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key in config.paths (created if missing)."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def sec_headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT") or load_config()["sec_user_agent"]
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
