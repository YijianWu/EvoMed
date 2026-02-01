"""Delta operations produced by the Engine Curator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional


# Extend operation types to support modular experience library
OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE", "UPDATE_MUTABLE"]


@dataclass
class DeltaOperation:
    """Single mutation to apply to the playbook."""

    type: OperationType
    section: str
    content: Optional[str] = None
    bullet_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Fields specific to modular experience library
    modules: Optional[Dict[str, Any]] = None  # Full module for ADD operation
    mutable_modules: Optional[Dict[str, Any]] = None  # Iterative module for UPDATE_MUTABLE operation

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DeltaOperation":
        # Parse metadata, handle mixed types
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
            modules=payload.get("modules"),  # Modular structure
            mutable_modules=payload.get("mutable_modules"),  # Iterative module update
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
