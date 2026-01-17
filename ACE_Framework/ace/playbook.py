"""Playbook storage and mutation logic for ACE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .delta import DeltaBatch, DeltaOperation


@dataclass
class Bullet:
    """Single playbook entry."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = 1) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Playbook:
    """Structured context store as defined by ACE."""

    def __init__(self) -> None:
        self._bullets: Dict[str, Bullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        self._stats_cache: Dict[str, object] = {
            "sections": 0,
            "bullets": 0,
            "tags": {"helpful": 0, "harmful": 0, "neutral": 0},
        }
        self._prompt_cache: str = ""
        self._prompt_dirty: bool = True
        self._bullet_index: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # CRUD utils
    # ------------------------------------------------------------------ #
    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        # 清洗 content，去掉可能被模型复制进来的计数字样
        clean_content = self._sanitize_content(content)
        bullet = Bullet(id=bullet_id, section=section, content=clean_content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        # update caches
        self._stats_cache["bullets"] = len(self._bullets)
        self._stats_cache["sections"] = len(self._sections)
        self._bullet_index[bullet_id] = bullet.content
        self._stats_cache["tags"]["helpful"] += bullet.helpful
        self._stats_cache["tags"]["harmful"] += bullet.harmful
        self._stats_cache["tags"]["neutral"] += bullet.neutral
        self._prompt_dirty = True
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            # 清洗 content，避免把渲染层的计数写入存储层
            new_content = self._sanitize_content(content)
            bullet.content = new_content
            # update index
            self._bullet_index[bullet_id] = new_content
        if metadata:
            # adjust tag counters diff
            before = {"helpful": bullet.helpful, "harmful": bullet.harmful, "neutral": bullet.neutral}
            bullet.apply_metadata(metadata)
            after = {"helpful": bullet.helpful, "harmful": bullet.harmful, "neutral": bullet.neutral}
            self._stats_cache["tags"]["helpful"] += (after["helpful"] - before["helpful"])
            self._stats_cache["tags"]["harmful"] += (after["harmful"] - before["harmful"]) 
            self._stats_cache["tags"]["neutral"] += (after["neutral"] - before["neutral"]) 
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        self._prompt_dirty = True
        return bullet

    def tag_bullet(self, bullet_id: str, tag: str, increment: int = 1) -> Optional[Bullet]:
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)
        # update stats counters
        if tag in self._stats_cache["tags"]:
            self._stats_cache["tags"][tag] += increment
        self._prompt_dirty = True
        return bullet

    def remove_bullet(self, bullet_id: str) -> None:
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [
                bid for bid in section_list if bid != bullet_id
            ]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]
        # update caches
        self._stats_cache["bullets"] = len(self._bullets)
        self._stats_cache["sections"] = len(self._sections)
        self._bullet_index.pop(bullet_id, None)
        self._stats_cache["tags"]["helpful"] -= bullet.helpful
        self._stats_cache["tags"]["harmful"] -= bullet.harmful
        self._stats_cache["tags"]["neutral"] -= bullet.neutral
        self._prompt_dirty = True

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    def get_bullet_content(self, bullet_id: str) -> Optional[str]:
        return self._bullet_index.get(bullet_id)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": {bullet_id: asdict(bullet) for bullet_id, bullet in self._bullets.items()},
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Playbook":
        instance = cls()
        bullets_payload = payload.get("bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    instance._bullets[bullet_id] = Bullet(**bullet_value)
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            instance._sections = {
                section: list(ids) if isinstance(ids, Iterable) else []
                for section, ids in sections_payload.items()
            }
        instance._next_id = int(payload.get("next_id", 0))
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "Playbook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook serialization must be a JSON object.")
        return cls.from_dict(payload)

    # ------------------------------------------------------------------ #
    # Delta application
    # ------------------------------------------------------------------ #
    def apply_delta(self, delta: DeltaBatch) -> None:
        for operation in delta.operations:
            self._apply_operation(operation)

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            self.add_bullet(
                section=operation.section,
                content=operation.content or "",
                bullet_id=operation.bullet_id,
                metadata=operation.metadata,
            )
        elif op_type == "UPDATE":
            if operation.bullet_id is None:
                return
            self.update_bullet(
                operation.bullet_id,
                content=operation.content,
                metadata=operation.metadata,
            )
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            for tag, increment in operation.metadata.items():
                self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #
    def as_prompt(self) -> str:
        """Return a human-readable playbook string for prompting LLMs.
        Uses a cached rendering; marks dirty on mutations.
        """
        if not self._prompt_dirty:
            return self._prompt_cache
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = (
                    f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                )
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        self._prompt_cache = "\n".join(parts)
        self._prompt_dirty = False
        return self._prompt_cache

    def stats(self) -> Dict[str, object]:
        return {
            "sections": self._stats_cache["sections"],
            "bullets": self._stats_cache["bullets"],
            "tags": {
                "helpful": self._stats_cache["tags"]["helpful"],
                "harmful": self._stats_cache["tags"]["harmful"],
                "neutral": self._stats_cache["tags"]["neutral"],
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower()
        return f"{section_prefix}-{self._next_id:05d}"

    def _sanitize_content(self, content: str) -> str:
        """Remove trailing tag counters like '(helpful=1, harmful=0[, neutral=0])'.

        仅移除末尾的计数字样，保留正文。大小写与空白均鲁棒。
        """
        if not isinstance(content, str) or not content:
            return content
        pattern = re.compile(
            r"\s*\((?:helpful|harmful|neutral)=\d+(?:,\s*(?:helpful|harmful|neutral)=\d+){0,2}\)\s*$",
            re.IGNORECASE,
        )
        cleaned = pattern.sub("", content)
        return cleaned.rstrip()
