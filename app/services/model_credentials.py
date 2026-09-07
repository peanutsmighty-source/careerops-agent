from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEPSEEK_KEY_FILE = PROJECT_ROOT / "ds_key.txt"


def load_deepseek_api_key() -> str:
    value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value

    configured_path = os.getenv("CAREEROPS_DEEPSEEK_KEY_FILE")
    path = Path(configured_path) if configured_path else DEFAULT_DEEPSEEK_KEY_FILE
    if not path.is_file():
        raise ValueError(
            "DeepSeek API key is not configured; set DEEPSEEK_API_KEY or create ds_key.txt"
        )
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("DeepSeek API key file is empty")
    if any(character.isspace() for character in value):
        raise ValueError("DeepSeek API key must not contain whitespace")
    return value
