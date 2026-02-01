"""Playbook storage and mutation logic for Engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .delta import DeltaBatch, DeltaOperation


# ------------------------------------------------------------------ #
# Modular Structure Definitions
# ------------------------------------------------------------------ #

@dataclass
class ContextualStates:
    """Fixed Module: Clinical contextual states (for vector retrieval)"""
    scenario: str = ""
    chief_complaint: str = ""
    core_symptoms: str = ""


@dataclass
class DecisionBehaviors:
    """Fixed Module: Diagnostic decision behaviors (for vector retrieval)"""
    diagnostic_path: str = ""


@dataclass
class Uncertainty:
    """Iterative Module: Diagnostic uncertainty (mutable in reflection phase)"""
    primary_uncertainty: str = ""


@dataclass
class DelayedAssumptions:
    """Iterative Module: Hypotheses to be verified (mutable in reflection phase)"""
    pending_validations: List[str] = field(default_factory=list)


@dataclass
class FixedModules:
    """Combination of fixed modules (for vector retrieval, immutable)"""
    contextual_states: ContextualStates = field(default_factory=ContextualStates)
    decision_behaviors: DecisionBehaviors = field(default_factory=DecisionBehaviors)

    def to_vector_text(self) -> str:
        parts = []
        if self.contextual_states.scenario:
            parts.append(f"Scenario: {self.contextual_states.scenario}")
        if self.contextual_states.chief_complaint:
            parts.append(f"Chief Complaint: {self.contextual_states.chief_complaint}")
        if self.contextual_states.core_symptoms:
            parts.append(f"Core Symptoms: {self.contextual_states.core_symptoms}")
        if self.decision_behaviors.diagnostic_path:
            parts.append(f"Diagnostic Path: {self.decision_behaviors.diagnostic_path}")
        return " | ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contextual_states": asdict(self.contextual_states),
            "decision_behaviors": asdict(self.decision_behaviors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FixedModules":
        instance = cls()
        if "contextual_states" in data and isinstance(data["contextual_states"], dict):
            instance.contextual_states = ContextualStates(**data["contextual_states"])
        if "decision_behaviors" in data and isinstance(data["decision_behaviors"], dict):
            instance.decision_behaviors = DecisionBehaviors(**data["decision_behaviors"])
        return instance


@dataclass
class MutableModules:
    """Combination of iterative modules (mutable in reflection phase)"""
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    delayed_assumptions: DelayedAssumptions = field(default_factory=DelayedAssumptions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainty": asdict(self.uncertainty),
            "delayed_assumptions": asdict(self.delayed_assumptions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutableModules":
        instance = cls()
        if "uncertainty" in data and isinstance(data["uncertainty"], dict):
            instance.uncertainty = Uncertainty(**data["uncertainty"])
        if "delayed_assumptions" in data and isinstance(data["delayed_assumptions"], dict):
            da_data = data["delayed_assumptions"]
            if isinstance(da_data.get("pending_validations"), list):
                instance.delayed_assumptions = DelayedAssumptions(
                    pending_validations=da_data["pending_validations"]
                )
        return instance


# ------------------------------------------------------------------ #
# Traditional Bullet Structure (Backward Compatibility)
# ------------------------------------------------------------------ #

@dataclass
class Bullet:
    """Single playbook entry."""
    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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


# ------------------------------------------------------------------ #
# Modular Bullet Structure
# ------------------------------------------------------------------ #

@dataclass
class ModularBullet:
    """Modular experience entry"""
    id: str
    section: str
    fixed_modules: FixedModules = field(default_factory=FixedModules)
    mutable_modules: MutableModules = field(default_factory=MutableModules)
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    evidence_level: str = ""
    reusable_score: float = 0.0
    clinical_impact: str = ""
    adaptability: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_vector_text(self) -> str:
        return self.fixed_modules.to_vector_text()

    def apply_metadata(self, metadata: Dict[str, Any]) -> None:
        for key, value in metadata.items():
            if hasattr(self, key):
                if key in ("helpful", "harmful", "neutral"):
                    setattr(self, key, int(value))
                elif key in ("reusable_score", "adaptability"):
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)

    def tag(self, tag: str, increment: int = 1) -> None:
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def update_mutable_modules(self, mutable_data: Dict[str, Any]) -> None:
        if "uncertainty" in mutable_data:
            unc_data = mutable_data["uncertainty"]
            if isinstance(unc_data, dict):
                self.mutable_modules.uncertainty.primary_uncertainty = unc_data.get(
                    "primary_uncertainty", self.mutable_modules.uncertainty.primary_uncertainty
                )
            elif isinstance(unc_data, str):
                self.mutable_modules.uncertainty.primary_uncertainty = unc_data
        if "delayed_assumptions" in mutable_data:
            da_data = mutable_data["delayed_assumptions"]
            if isinstance(da_data, dict) and "pending_validations" in da_data:
                self.mutable_modules.delayed_assumptions.pending_validations = da_data["pending_validations"]
            elif isinstance(da_data, list):
                self.mutable_modules.delayed_assumptions.pending_validations = da_data
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "fixed_modules": self.fixed_modules.to_dict(),
            "mutable_modules": self.mutable_modules.to_dict(),
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
            "evidence_level": self.evidence_level,
            "reusable_score": self.reusable_score,
            "clinical_impact": self.clinical_impact,
            "adaptability": self.adaptability,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModularBullet":
        fixed = FixedModules.from_dict(data.get("fixed_modules", {}))
        mutable = MutableModules.from_dict(data.get("mutable_modules", {}))
        return cls(
            id=data.get("id", ""),
            section=data.get("section", ""),
            fixed_modules=fixed,
            mutable_modules=mutable,
            helpful=int(data.get("helpful", 0)),
            harmful=int(data.get("harmful", 0)),
            neutral=int(data.get("neutral", 0)),
            evidence_level=data.get("evidence_level", ""),
            reusable_score=float(data.get("reusable_score", 0.0)),
            clinical_impact=data.get("clinical_impact", ""),
            adaptability=float(data.get("adaptability", 0.0)),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )

    def as_prompt_text(self) -> str:
        lines = []
        if self.fixed_modules.contextual_states.scenario:
            lines.append(f"  Scenario: {self.fixed_modules.contextual_states.scenario}")
        if self.fixed_modules.contextual_states.chief_complaint:
            lines.append(f"  Chief Complaint: {self.fixed_modules.contextual_states.chief_complaint}")
        if self.fixed_modules.contextual_states.core_symptoms:
            lines.append(f"  Core Symptoms: {self.fixed_modules.contextual_states.core_symptoms}")
        if self.fixed_modules.decision_behaviors.diagnostic_path:
            lines.append(f"  Diagnostic Path: {self.fixed_modules.decision_behaviors.diagnostic_path}")
        if self.mutable_modules.uncertainty.primary_uncertainty:
            lines.append(f"  Uncertainty: {self.mutable_modules.uncertainty.primary_uncertainty}")
        if self.mutable_modules.delayed_assumptions.pending_validations:
            validations = ", ".join(self.mutable_modules.delayed_assumptions.pending_validations)
            lines.append(f"  To be verified: {validations}")
        return "\n".join(lines) if lines else "(empty)"


# ------------------------------------------------------------------ #
# Traditional Playbook Class
# ------------------------------------------------------------------ #

class Playbook:
    """Structured context store as defined by Engine."""

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

    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        clean_content = self._sanitize_content(content)
        bullet = Bullet(id=bullet_id, section=section, content=clean_content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
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
            new_content = self._sanitize_content(content)
            bullet.content = new_content
            self._bullet_index[bullet_id] = new_content
        if metadata:
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
            self._sections[bullet.section] = [bid for bid in section_list if bid != bullet_id]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]
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
            self.update_bullet(operation.bullet_id, content=operation.content, metadata=operation.metadata)
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            for tag, increment in operation.metadata.items():
                self.tag_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_bullet(operation.bullet_id)

    def as_prompt(self) -> str:
        if not self._prompt_dirty:
            return self._prompt_cache
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._bullets[bullet_id]
                counters = f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
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

    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower() if section else "bullet"
        return f"{section_prefix}-{self._next_id:05d}"

    def _sanitize_content(self, content: str) -> str:
        if not isinstance(content, str) or not content:
            return content
        pattern = re.compile(
            r"\s*\((?:helpful|harmful|neutral)=\d+(?:,\s*(?:helpful|harmful|neutral)=\d+){0,2}\)\s*$",
            re.IGNORECASE,
        )
        cleaned = pattern.sub("", content)
        return cleaned.rstrip()


# ------------------------------------------------------------------ #
# Modular Playbook Class
# ------------------------------------------------------------------ #

class ModularPlaybook:
    """Modular experience library: supports fixed modules (vector retrieval) and iterative modules (reflection modification)"""

    def __init__(self) -> None:
        self._modular_bullets: Dict[str, ModularBullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0
        self._stats_cache: Dict[str, object] = {
            "sections": 0,
            "bullets": 0,
            "tags": {"helpful": 0, "harmful": 0, "neutral": 0},
        }
        self._prompt_cache: str = ""
        self._prompt_dirty: bool = True
        self._vector_text_index: Dict[str, str] = {}

    def add_modular_bullet(
        self,
        section: str,
        fixed_modules: Dict[str, Any],
        mutable_modules: Optional[Dict[str, Any]] = None,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModularBullet:
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        mutable_modules = mutable_modules or {}

        fixed = FixedModules()
        if "contextual_states" in fixed_modules:
            cs = fixed_modules["contextual_states"]
            fixed.contextual_states = ContextualStates(
                scenario=cs.get("scenario", ""),
                chief_complaint=cs.get("chief_complaint", ""),
                core_symptoms=cs.get("core_symptoms", ""),
            )
        if "decision_behaviors" in fixed_modules:
            db = fixed_modules["decision_behaviors"]
            fixed.decision_behaviors = DecisionBehaviors(
                diagnostic_path=db.get("diagnostic_path", ""),
            )

        mutable = MutableModules()
        if "uncertainty" in mutable_modules:
            unc = mutable_modules["uncertainty"]
            if isinstance(unc, dict):
                mutable.uncertainty = Uncertainty(primary_uncertainty=unc.get("primary_uncertainty", ""))
            elif isinstance(unc, str):
                mutable.uncertainty = Uncertainty(primary_uncertainty=unc)
        if "delayed_assumptions" in mutable_modules:
            da = mutable_modules["delayed_assumptions"]
            if isinstance(da, dict):
                mutable.delayed_assumptions = DelayedAssumptions(pending_validations=da.get("pending_validations", []))
            elif isinstance(da, list):
                mutable.delayed_assumptions = DelayedAssumptions(pending_validations=da)

        bullet = ModularBullet(id=bullet_id, section=section, fixed_modules=fixed, mutable_modules=mutable)
        bullet.apply_metadata(metadata)

        self._modular_bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        self._stats_cache["bullets"] = len(self._modular_bullets)
        self._stats_cache["sections"] = len(self._sections)
        self._vector_text_index[bullet_id] = bullet.get_vector_text()
        self._stats_cache["tags"]["helpful"] += bullet.helpful
        self._stats_cache["tags"]["harmful"] += bullet.harmful
        self._stats_cache["tags"]["neutral"] += bullet.neutral
        self._prompt_dirty = True
        return bullet

    def update_mutable_modules(self, bullet_id: str, mutable_data: Dict[str, Any]) -> Optional[ModularBullet]:
        bullet = self._modular_bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.update_mutable_modules(mutable_data)
        self._prompt_dirty = True
        return bullet

    def tag_modular_bullet(self, bullet_id: str, tag: str, increment: int = 1) -> Optional[ModularBullet]:
        bullet = self._modular_bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment=increment)
        if tag in self._stats_cache["tags"]:
            self._stats_cache["tags"][tag] += increment
        self._prompt_dirty = True
        return bullet

    def get_modular_bullet(self, bullet_id: str) -> Optional[ModularBullet]:
        return self._modular_bullets.get(bullet_id)

    def modular_bullets(self) -> List[ModularBullet]:
        return list(self._modular_bullets.values())

    def get_vector_text(self, bullet_id: str) -> Optional[str]:
        return self._vector_text_index.get(bullet_id)

    def remove_modular_bullet(self, bullet_id: str) -> None:
        bullet = self._modular_bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self._sections.get(bullet.section)
        if section_list:
            self._sections[bullet.section] = [bid for bid in section_list if bid != bullet_id]
            if not self._sections[bullet.section]:
                del self._sections[bullet.section]
        self._stats_cache["bullets"] = len(self._modular_bullets)
        self._stats_cache["sections"] = len(self._sections)
        self._vector_text_index.pop(bullet_id, None)
        self._stats_cache["tags"]["helpful"] -= bullet.helpful
        self._stats_cache["tags"]["harmful"] -= bullet.harmful
        self._stats_cache["tags"]["neutral"] -= bullet.neutral
        self._prompt_dirty = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "modular_bullets": {
                bullet_id: bullet.to_dict()
                for bullet_id, bullet in self._modular_bullets.items()
            },
            "sections": self._sections,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ModularPlaybook":
        instance = cls()
        bullets_payload = payload.get("modular_bullets", {})
        if isinstance(bullets_payload, dict):
            for bullet_id, bullet_value in bullets_payload.items():
                if isinstance(bullet_value, dict):
                    bullet = ModularBullet.from_dict(bullet_value)
                    instance._modular_bullets[bullet_id] = bullet
                    instance._vector_text_index[bullet_id] = bullet.get_vector_text()
        sections_payload = payload.get("sections", {})
        if isinstance(sections_payload, dict):
            instance._sections = {
                section: list(ids) if isinstance(ids, Iterable) else []
                for section, ids in sections_payload.items()
            }
        instance._next_id = int(payload.get("next_id", 0))
        instance._stats_cache["bullets"] = len(instance._modular_bullets)
        instance._stats_cache["sections"] = len(instance._sections)
        return instance

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def loads(cls, data: str) -> "ModularPlaybook":
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("ModularPlaybook serialization must be a JSON object.")
        return cls.from_dict(payload)

    def apply_delta(self, delta: DeltaBatch) -> None:
        for operation in delta.operations:
            self._apply_operation(operation)

    def _apply_operation(self, operation: DeltaOperation) -> None:
        op_type = operation.type.upper()
        if op_type == "ADD":
            modules = getattr(operation, "modules", None) or {}
            fixed_modules = {
                "contextual_states": modules.get("contextual_states", {}),
                "decision_behaviors": modules.get("decision_behaviors", {}),
            }
            mutable_modules = {
                "uncertainty": modules.get("uncertainty", {}),
                "delayed_assumptions": modules.get("delayed_assumptions", {}),
            }
            self.add_modular_bullet(
                section=operation.section,
                fixed_modules=fixed_modules,
                mutable_modules=mutable_modules,
                bullet_id=operation.bullet_id,
                metadata=operation.metadata,
            )
        elif op_type == "UPDATE_MUTABLE":
            if operation.bullet_id is None:
                return
            mutable_data = getattr(operation, "mutable_modules", None) or {}
            self.update_mutable_modules(operation.bullet_id, mutable_data)
        elif op_type == "TAG":
            if operation.bullet_id is None:
                return
            for tag, increment in operation.metadata.items():
                self.tag_modular_bullet(operation.bullet_id, tag, increment)
        elif op_type == "REMOVE":
            if operation.bullet_id is None:
                return
            self.remove_modular_bullet(operation.bullet_id)

    def as_prompt(self) -> str:
        if not self._prompt_dirty:
            return self._prompt_cache
        parts: List[str] = []
        for section, bullet_ids in sorted(self._sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self._modular_bullets[bullet_id]
                counters = f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                parts.append(f"- [{bullet.id}]")
                parts.append(bullet.as_prompt_text())
                parts.append(f"  {counters}")
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

    def _generate_id(self, section: str) -> str:
        self._next_id += 1
        section_prefix = section.split()[0].lower() if section else "mod"
        return f"{section_prefix}-{self._next_id:05d}"
