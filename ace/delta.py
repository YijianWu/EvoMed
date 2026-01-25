"""Delta operations produced by the ACE Curator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional


# 扩展操作类型以支持模块化经验库
OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE", "UPDATE_MUTABLE"]


@dataclass
class DeltaOperation:
    """Single mutation to apply to the playbook."""

    type: OperationType
    section: str
    content: Optional[str] = None
    bullet_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 模块化经验库专用字段
    modules: Optional[Dict[str, Any]] = None  # ADD 操作的完整模块
    mutable_modules: Optional[Dict[str, Any]] = None  # UPDATE_MUTABLE 操作的可迭代模块

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DeltaOperation":
        # 解析 metadata，处理混合类型
        raw_metadata = payload.get("metadata") or {}
        metadata = {}
        if isinstance(raw_metadata, dict):
            for k, v in raw_metadata.items():
                if isinstance(v, (int, float)):
                    metadata[str(k)] = v
                elif isinstance(v, str):
                    metadata[str(k)] = v
                else:
                    metadata[str(k)] = v

        return cls(
            type=str(payload["type"]),
            section=str(payload.get("section", "")),
            content=payload.get("content") and str(payload["content"]),
            bullet_id=payload.get("bullet_id")
            and str(payload.get("bullet_id")),  # type: ignore[arg-type]
            metadata=metadata,
            modules=payload.get("modules"),  # 模块化结构
            mutable_modules=payload.get("mutable_modules"),  # 可迭代模块更新
        )

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {"type": self.type, "section": self.section}
        if self.content is not None:
            data["content"] = self.content
        if self.bullet_id is not None:
            data["bullet_id"] = self.bullet_id
        if self.metadata:
            data["metadata"] = self.metadata
        if self.modules is not None:
            data["modules"] = self.modules
        if self.mutable_modules is not None:
            data["mutable_modules"] = self.mutable_modules
        return data


@dataclass
class DeltaBatch:
    """Bundle of curator reasoning and operations."""

    reasoning: str
    operations: List[DeltaOperation] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DeltaBatch":
        ops_payload = payload.get("operations")
        operations = []
        if isinstance(ops_payload, Iterable):
            for item in ops_payload:
                if isinstance(item, dict):
                    operations.append(DeltaOperation.from_json(item))
        return cls(reasoning=str(payload.get("reasoning", "")), operations=operations)

    def to_json(self) -> Dict[str, object]:
        return {
            "reasoning": self.reasoning,
            "operations": [op.to_json() for op in self.operations],
        }
