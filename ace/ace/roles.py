"""Generator, Reflector, and Curator components."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .delta import DeltaBatch
from .llm import LLMClient
from .playbook import Playbook
from .prompts import CURATOR_PROMPT, MODULAR_REFLECTOR_PROMPT, REFLECTOR_PROMPT


# ---------------------------------------------------------------------
# Utility: robust JSON parsing for Reflector / Curator LLM outputs
# ---------------------------------------------------------------------
def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to robustly parse JSON returned by the model:
    - Remove markdown code blocks ```json ... ```
    - Extract the first {...} JSON object in the body
    - Write raw text to logs/json_failures.log if parsing fails
    """
    original_text = text
    text = text.strip()

    # If it starts with markdown ```json, remove the fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    # If it doesn't start with {, try to grab the first { ... } block
    if not text.strip().startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()

    # Do the actual json.loads
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write("RAW RESPONSE:\n")
            fh.write(repr(original_text))
            fh.write("\n\nEXTRACTED:\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(
            f"LLM response is not valid JSON: {exc}\nExtracted:\n{text}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object from LLM, got {type(data)}")

    return data


def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"


# ---------------------------------------------------------------------
# Generator Output
# ---------------------------------------------------------------------

@dataclass
class GeneratorOutput:
    """
    Results of the diagnosis stage (passed in from external system).
    """
    reasoning: str         # Our diagnostic rationale
    final_answer: str      # Our final diagnosis (most likely diagnosis)
    bullet_ids: List[str]  # Not used for now, defaulted to []
    raw: Dict[str, Any]    # Raw full information, accessible by Reflector / Curator later


# ---------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------

@dataclass
class BulletTag:
    id: str
    tag: str  # "helpful" / "harmful" / "neutral"
    reason: str = ""


@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]


class Reflector:
    """
    Reflector:
    - Looks at GeneratorOutput (your diagnosis + explanation)
    - Looks at ground_truth / feedback (environment score)
    - Combined with the current Playbook, requests the LLM to produce a structured JSON reflection
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        playbook_excerpt: Optional[str] = None,
        allowed_ids: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        # If external provide concurrency-safe snapshot, use it; otherwise fall back to global object reading.
        if playbook_excerpt is None:
            playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt or "(no bullets referenced)",
            allowed_ids=allowed_ids or "",
        )

        result: Optional[ReflectorOutput] = None
        last_error: Optional[Exception] = None

        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt,
                    refinement_round=round_idx,
                    **kwargs,
                )
                try:
                    data = _safe_json_loads(response.text)

                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if (
                                isinstance(item, dict)
                                and "id" in item
                                and "tag" in item
                            ):
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]),
                                        tag=str(item["tag"]).lower(),
                                        reason=str(item.get("reason", "")),
                                    )
                                )

                    if isinstance(allowed_ids, list):
                        allowed_set = set(allowed_ids)
                    else:
                        allowed_set = set(s.strip() for s in (allowed_ids or "").split(",") if s.strip())
                    if allowed_set:
                        bullet_tags = [bt for bt in bullet_tags if bt.id in allowed_set]

                    candidate = ReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        raw=data,
                    )
                    result = candidate

                    # If it already gave usable bullet_tags or key_insight, can stop refinement early
                    if bullet_tags or candidate.key_insight:
                        return candidate

                    break  # Jump out of attempt loop if no error

                except ValueError as err:
                    last_error = err
                    # Retry prompt to force strict JSON output
                    if attempt + 1 >= self.max_retries:
                        break
                    base_prompt = (
                        base_prompt
                        + "\n\nPlease strictly output valid JSON, escape double quotes, "
                          "and do not output extra explanatory text."
                    )

        if result is None:
            raise RuntimeError(
                "Reflector failed to produce a result."
            ) from last_error
        return result


# ---------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------

@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """
    Curator:
    - Looks at Reflector's reflection (including key mistakes, key insights)
    - Looks at current Playbook
    - Decides whether to ADD / UPDATE / TAG / REMOVE bullet points in the Playbook
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        playbook_text: Optional[str] = None,
        playbook_stats: Optional[str] = None,
        **kwargs: Any,
    ) -> CuratorOutput:
        # If external provide concurrency-safe snapshot, use it; otherwise fall back to global object reading.
        stats_value = playbook_stats if playbook_stats is not None else json.dumps(playbook.stats(), ensure_ascii=False)
        playbook_value = playbook_text if playbook_text is not None else (playbook.as_prompt() or "(empty playbook)")
        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=stats_value,
            reflection=json.dumps(
                reflection.raw,
                ensure_ascii=False,
                indent=2,
            ),
            playbook=playbook_value,
            question_context=question_context,
        )

        last_error: Optional[Exception] = None
        prompt = base_prompt

        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                delta = DeltaBatch.from_json(data)
                if not reflection.bullet_tags:
                    delta.operations = [op for op in delta.operations if op.type.upper() != "TAG"]
                    try:
                        ops_payload = data.get("operations")
                        if isinstance(ops_payload, list):
                            data["operations"] = [
                                item for item in ops_payload
                                if not (isinstance(item, dict) and str(item.get("type", "")).upper() == "TAG")
                            ]
                    except Exception:
                        pass
                return CuratorOutput(delta=delta, raw=data)

            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                # Retry prompt
                prompt = (
                    base_prompt
                    + "\n\nReminder: Only output valid JSON, escape all double quotes in strings or use single quotes, "
                      "and do not add extra text."
                )

        raise RuntimeError(
            "Curator failed to produce valid JSON."
        ) from last_error


# ---------------------------------------------------------------------
# Helper: build little excerpt of the Playbook bullets the Generator said it used
# ---------------------------------------------------------------------
def _make_playbook_excerpt(playbook: Playbook, bullet_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Modular Reflector: Supports reflection of modular experience and iterative module updates
# ---------------------------------------------------------------------

@dataclass
class MutableUpdate:
    """Iterative module update"""
    bullet_id: str
    uncertainty: Optional[Dict[str, str]] = None
    delayed_assumptions: Optional[Dict[str, List[str]]] = None


@dataclass
class ModularReflectorOutput:
    """Modular reflection output"""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    mutable_updates: List[MutableUpdate]  # List of iterative module updates
    raw: Dict[str, Any]

    def to_reflector_output(self) -> ReflectorOutput:
        """Convert to traditional ReflectorOutput for compatibility with Curator"""
        return ReflectorOutput(
            reasoning=self.reasoning,
            error_identification=self.error_identification,
            root_cause_analysis=self.root_cause_analysis,
            correct_approach=self.correct_approach,
            key_insight=self.key_insight,
            bullet_tags=self.bullet_tags,
            raw=self.raw,
        )


class ModularReflector:
    """
    Modular Reflector:
    - Evaluates retrieved modular experiences
    - Generates update suggestions for iterative modules (UPDATE_MUTABLE operation)
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = MODULAR_REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        modular_excerpts: str,  # Modular experience summary
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        allowed_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ModularReflectorOutput:
        """
        Execute modular reflection

        Args:
            question: Diagnostic question
            generator_output: Generator output
            modular_excerpts: Modular experience summary (containing fixed and iterative modules)
            ground_truth: Standard answer
            feedback: Environment feedback
            max_refinement_rounds: Maximum refinement rounds
            allowed_ids: List of bullet_ids allowed for evaluation
        """
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            modular_excerpts=modular_excerpts or "(no modular experiences retrieved)",
        )

        result: Optional[ModularReflectorOutput] = None
        last_error: Optional[Exception] = None

        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt,
                    refinement_round=round_idx,
                    **kwargs,
                )
                try:
                    data = _safe_json_loads(response.text)

                    # Parse bullet_tags
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if isinstance(item, dict) and "id" in item and "tag" in item:
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]),
                                        tag=str(item["tag"]).lower(),
                                        reason=str(item.get("reason", "")),
                                    )
                                )

                    # Filter allowed_ids
                    if allowed_ids:
                        allowed_set = set(allowed_ids)
                        bullet_tags = [bt for bt in bullet_tags if bt.id in allowed_set]

                    # Parse mutable_updates
                    mutable_updates: List[MutableUpdate] = []
                    updates_payload = data.get("mutable_updates", [])
                    if isinstance(updates_payload, Sequence):
                        for item in updates_payload:
                            if isinstance(item, dict) and "bullet_id" in item:
                                update = MutableUpdate(
                                    bullet_id=str(item["bullet_id"]),
                                    uncertainty=item.get("uncertainty"),
                                    delayed_assumptions=item.get("delayed_assumptions"),
                                )
                                # Filter allowed_ids
                                if allowed_ids and update.bullet_id not in allowed_ids:
                                    continue
                                mutable_updates.append(update)

                    candidate = ModularReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        mutable_updates=mutable_updates,
                        raw=data,
                    )
                    result = candidate

                    if bullet_tags or candidate.key_insight or mutable_updates:
                        return candidate

                    break

                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    base_prompt = (
                        base_prompt
                        + "\n\nPlease strictly output valid JSON, escape double quotes, and do not output extra explanatory text."
                    )

        if result is None:
            raise RuntimeError("ModularReflector failed to produce a result.") from last_error
        return result


def build_modular_excerpt(
    retrieved_results: List[Any],  # List[ModularRetrievalResult]
) -> str:
    """
    Build modular experience summary for reflection stage

    Args:
        retrieved_results: List of modular retrieval results

    Returns:
        Formatted experience summary text
    """
    if not retrieved_results:
        return "(no modular experiences retrieved)"

    lines = []
    for i, result in enumerate(retrieved_results, 1):
        lines.append(f"=== Experience {i} [ID: {result.bullet_id}] ===")
        lines.append(f"Section: {result.section}")
        lines.append("")
        
        # Fixed module (immutable, used for matching)
        lines.append("【Fixed Module - for retrieval matching】")
        fixed = result.fixed_modules
        cs = fixed.get("contextual_states", {})
        if cs.get("scenario"):
            lines.append(f"  Scenario: {cs['scenario']}")
        if cs.get("chief_complaint"):
            lines.append(f"  Chief Complaint: {cs['chief_complaint']}")
        if cs.get("core_symptoms"):
            lines.append(f"  Core Symptoms: {cs['core_symptoms']}")
        db = fixed.get("decision_behaviors", {})
        if db.get("diagnostic_path"):
            lines.append(f"  Diagnostic Path: {db['diagnostic_path']}")
        lines.append("")
        
        # Iterative module (mutable)
        lines.append("【Iterative Module - can be updated in reflection】")
        mutable = result.mutable_modules
        unc = mutable.get("uncertainty", {})
        if isinstance(unc, dict) and unc.get("primary_uncertainty"):
            lines.append(f"  Current Uncertainty: {unc['primary_uncertainty']}")
        elif isinstance(unc, str) and unc:
            lines.append(f"  Current Uncertainty: {unc}")
        else:
            lines.append("  Current Uncertainty: (to be added)")
        
        da = mutable.get("delayed_assumptions", {})
        if isinstance(da, dict) and da.get("pending_validations"):
            validations = ", ".join(da["pending_validations"])
            lines.append(f"  Hypotheses to be verified: {validations}")
        elif isinstance(da, list) and da:
            lines.append(f"  Hypotheses to be verified: {', '.join(da)}")
        else:
            lines.append("  Hypotheses to be verified: (to be added)")
        
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
