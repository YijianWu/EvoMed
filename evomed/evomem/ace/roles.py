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
from .prompts import CURATOR_PROMPT, GENERATOR_PROMPT, REFLECTOR_PROMPT


# ---------------------------------------------------------------------
# Utility: robust JSON parsing for Reflector / Curator LLM outputs
# ---------------------------------------------------------------------
def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    尝试鲁棒地解析模型返回的 JSON：
    - 去掉 markdown 代码块 ```json ... ```
    - 提取正文中第一个 {...} JSON 对象
    - 解析失败会把原始文本写入 logs/json_failures.log
    """
    original_text = text
    text = text.strip()

    # 如果是 markdown ```json 开头，去掉围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    # 如果不是 { 开头，尝试抓第一个 { ... } 块
    if not text.strip().startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()

    # 做真正的 json.loads
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
# Generator
# ---------------------------------------------------------------------

@dataclass
class GeneratorOutput:
    """
    生成阶段的结果（无论它是LLM推出来的，还是我们外部系统喂进来的）。
    """
    reasoning: str         # 我们的诊断依据 diagnostic_rationale
    final_answer: str      # 我们的最终诊断 most_likely_diagnosis
    bullet_ids: List[str]  # 这次用不上，先给 []
    raw: Dict[str, Any]    # 原始全量信息，后续 Reflector / Curator 还能看到


class Generator:
    """
    临床定制版 Generator：
    - 不再调用 LLM。
    - 直接使用我们已经有的结构化诊断结果。
    - reasoning  = diagnostic_rationale
    - final_answer = most_likely_diagnosis
    - bullet_ids：暂时使用传进来的，或者空列表。
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        prompt_template: str = GENERATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        # 为了兼容原有接口，参数依旧保留，但不会真正用 llm 生成
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        我们不再根据 prompt 调 LLM，而是直接吃 kwargs 里的内容。
        这些 kwargs 会由 AdapterBase._process_sample() 通过 sample.metadata 传进来。

        需要的 keys:
          - most_likely_diagnosis      (string)
          - diagnostic_rationale      (string)
          - bullet_ids                (list[str], 可选)
        """

        most_likely = kwargs.get("most_likely_diagnosis", "")
        rationale   = kwargs.get("diagnostic_rationale", "")
        bullet_ids = kwargs.get("bullet_ids") or kwargs.get("retrieved_bullet_ids") or []

        # 类型兜底，避免不是字符串/不是列表时报错
        if not isinstance(most_likely, str):
            most_likely = str(most_likely)
        if not isinstance(rationale, str):
            rationale = str(rationale)
        if not isinstance(bullet_ids, list):
            bullet_ids = []

        raw_data: Dict[str, Any] = {
            "most_likely_diagnosis": most_likely,
            "diagnostic_rationale": rationale,
            "question": question,
            "context": context,
        }

        return GeneratorOutput(
            reasoning=rationale,
            final_answer=most_likely,
            bullet_ids=bullet_ids,
            raw=raw_data,
        )


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
    反思者：
    - 看 GeneratorOutput（你的诊断 + 解释）
    - 看 ground_truth / feedback（环境评分）
    - 结合当前 Playbook，要求 LLM 产出 structured JSON 反思
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
        # 如果外部提供了并发安全的快照，则直接使用；否则退回到基于全局对象的读取。
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

                    # 如果它已经给了可用的 bullet_tags 或 key_insight，就可以提前停止 refinement
                    if bullet_tags or candidate.key_insight:
                        return candidate

                    break  # 没报错就跳出 attempt 循环

                except ValueError as err:
                    last_error = err
                    # 重试提示，逼它输出严格 JSON
                    if attempt + 1 >= self.max_retries:
                        break
                    base_prompt = (
                        base_prompt
                        + "\n\n请严格输出有效 JSON，对双引号进行转义，"
                          "不要输出额外解释性文本。"
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
    策展者：
    - 看 Reflector 的反思（包括关键失误、关键启发）
    - 看当前 Playbook
    - 决定要不要往 Playbook 里 ADD / UPDATE / TAG / REMOVE 子弹点
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
        # 如果外部提供了并发安全的快照，则直接使用；否则退回到基于全局对象的读取。
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
                # 重试提示
                prompt = (
                    base_prompt
                    + "\n\n提醒：仅输出有效 JSON，所有字符串请转义双引号或改用单引号，"
                      "不要添加额外文本。"
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
