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


# ---------------------------------------------------------------------
# Modular Reflector: 支持模块化经验的反思和可迭代模块更新
# ---------------------------------------------------------------------

MODULAR_REFLECTOR_PROMPT = """\
你是一名资深临床医学专家，正在分析该病例的诊断过程。
你的任务是：
1. 评估检索到的经验对诊断的帮助程度
2. 基于当前病例的诊断经验，**更新检索到经验的可迭代模块**

Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}

检索到的相关经验（需要评估和更新）：
{modular_excerpts}

**重要任务：**
1. 对每个检索到的经验进行评估（helpful/harmful/neutral）
2. 如果当前病例提供了新的诊断不确定性或待验证假设，则更新对应经验的可迭代模块

Return JSON:
{{
  "reasoning": "<诊断过程分析>",
  "key_diagnostic_info": "<核心诊断要素>",
  "diagnostic_reasoning_path": "<诊断推理路径>",
  "correct_approach": "<正确诊断方法>",
  "key_insight": "<可复用的医学洞见>",
  "bullet_tags": [
    {{"id": "<bullet_id>", "tag": "helpful|harmful|neutral", "reason": "<简短原因>"}}
  ],
  "mutable_updates": [
    {{
      "bullet_id": "<要更新的经验ID>",
      "uncertainty": {{
        "primary_uncertainty": "<更新后的诊断不确定性，如不更新则省略此字段>"
      }},
      "delayed_assumptions": {{
        "pending_validations": ["<更新后的待验证假设列表>"]
      }}
    }}
  ]
}}

注意：
- mutable_updates 仅包含需要更新的经验，如无需更新则返回空数组 []
- 更新应基于当前病例的诊断经验，补充或修正原有的不确定性和待验证假设
- 保留有价值的原始内容，仅添加或修正需要改进的部分
"""


@dataclass
class MutableUpdate:
    """可迭代模块更新"""
    bullet_id: str
    uncertainty: Optional[Dict[str, str]] = None
    delayed_assumptions: Optional[Dict[str, List[str]]] = None


@dataclass
class ModularReflectorOutput:
    """模块化反思输出"""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    mutable_updates: List[MutableUpdate]  # 可迭代模块更新列表
    raw: Dict[str, Any]

    def to_reflector_output(self) -> ReflectorOutput:
        """转换为传统 ReflectorOutput 以兼容 Curator"""
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
    模块化反思器：
    - 评估检索到的模块化经验
    - 生成可迭代模块的更新建议（UPDATE_MUTABLE 操作）
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
        modular_excerpts: str,  # 模块化经验摘要
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        allowed_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ModularReflectorOutput:
        """
        执行模块化反思

        Args:
            question: 诊断问题
            generator_output: 生成器输出
            modular_excerpts: 模块化经验摘要（包含固定和可迭代模块）
            ground_truth: 标准答案
            feedback: 环境反馈
            max_refinement_rounds: 最大精炼轮数
            allowed_ids: 允许评估的 bullet_id 列表
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

                    # 解析 bullet_tags
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

                    # 过滤 allowed_ids
                    if allowed_ids:
                        allowed_set = set(allowed_ids)
                        bullet_tags = [bt for bt in bullet_tags if bt.id in allowed_set]

                    # 解析 mutable_updates
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
                                # 过滤 allowed_ids
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
                        + "\n\n请严格输出有效 JSON，对双引号进行转义，不要输出额外解释性文本。"
                    )

        if result is None:
            raise RuntimeError("ModularReflector failed to produce a result.") from last_error
        return result


def build_modular_excerpt(
    retrieved_results: List[Any],  # List[ModularRetrievalResult]
) -> str:
    """
    构建模块化经验摘要，用于反思阶段

    Args:
        retrieved_results: 模块化检索结果列表

    Returns:
        格式化的经验摘要文本
    """
    if not retrieved_results:
        return "(no modular experiences retrieved)"

    lines = []
    for i, result in enumerate(retrieved_results, 1):
        lines.append(f"=== 经验 {i} [ID: {result.bullet_id}] ===")
        lines.append(f"章节: {result.section}")
        lines.append("")
        
        # 固定模块（不可修改，用于匹配）
        lines.append("【固定模块 - 用于检索匹配】")
        fixed = result.fixed_modules
        cs = fixed.get("contextual_states", {})
        if cs.get("scenario"):
            lines.append(f"  情境: {cs['scenario']}")
        if cs.get("chief_complaint"):
            lines.append(f"  主诉: {cs['chief_complaint']}")
        if cs.get("core_symptoms"):
            lines.append(f"  核心症状: {cs['core_symptoms']}")
        db = fixed.get("decision_behaviors", {})
        if db.get("diagnostic_path"):
            lines.append(f"  诊断路径: {db['diagnostic_path']}")
        lines.append("")
        
        # 可迭代模块（可修改）
        lines.append("【可迭代模块 - 可在反思中更新】")
        mutable = result.mutable_modules
        unc = mutable.get("uncertainty", {})
        if isinstance(unc, dict) and unc.get("primary_uncertainty"):
            lines.append(f"  当前不确定性: {unc['primary_uncertainty']}")
        elif isinstance(unc, str) and unc:
            lines.append(f"  当前不确定性: {unc}")
        else:
            lines.append("  当前不确定性: (待补充)")
        
        da = mutable.get("delayed_assumptions", {})
        if isinstance(da, dict) and da.get("pending_validations"):
            validations = ", ".join(da["pending_validations"])
            lines.append(f"  待验证假设: {validations}")
        elif isinstance(da, list) and da:
            lines.append(f"  待验证假设: {', '.join(da)}")
        else:
            lines.append("  待验证假设: (待补充)")
        
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
