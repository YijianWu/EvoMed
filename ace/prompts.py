"""Prompt templates - imports from the root prompts.py file."""

import sys
from pathlib import Path

# 确保根目录在 Python 路径中
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 从根目录的 prompts.py 导入（使用新的原版 prompt）
from prompts import (
    GENERATOR_PROMPT,
    REFLECTOR_PROMPT,
    CURATOR_PROMPT,
)

__all__ = [
    "GENERATOR_PROMPT",
    "REFLECTOR_PROMPT",
    "CURATOR_PROMPT",
]
