#!/usr/bin/env python3
"""Run ACE adaptation on the sample questions and generate a report.

Version 3: 统一经验库版本，所有样本都进入ACE反思流程。
从 Excel 的 retrieved_experiences_json 列提取 bullet_id。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List
import pandas as pd  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (  # noqa: E402
    AdapterStepResult,
    Curator,
    EnvironmentResult,
    Generator,
    OfflineAdapter,
    Playbook,
    Reflector,
    Sample,
    TaskEnvironment,
    UniversalLLMClient,
)

def load_playbook_from_report(path: str) -> Playbook:
    """
    从 questions_report_*.md 文件中自动抽取 "## Final Playbook" 段落，
    并解析成 Playbook 对象。
    """
    pb = Playbook()
    current_section = None
    in_playbook = False

    if not os.path.exists(path):
        print(f"[WARN] report not found: {path}")
        return pb

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            # 找到 Final Playbook 开始
            if line.strip() == "## Final Playbook":
                in_playbook = True
                continue

            if not in_playbook:
                continue

            # 新 Section
            if line.startswith("## "):
                current_section = line[3:].strip()
                continue

            # Bullet 解析： - [ID] content (helpful=H, harmful=R[, neutral=N])
            m = re.match(r"- \[([^\]]+)\]\s+(.*)", line)
            if m:
                bullet_id = m.group(1).strip()
                content_full = m.group(2).strip()
                meta: Dict[str, int] = {}
                # 提取尾部计数
                m_meta = re.search(
                    r"\(helpful=(\d+),\s*harmful=(\d+)(?:,\s*neutral=(\d+))?\)\s*$",
                    content_full,
                    flags=re.IGNORECASE,
                )
                if m_meta:
                    meta["helpful"] = int(m_meta.group(1))
                    meta["harmful"] = int(m_meta.group(2))
                    if m_meta.group(3):
                        meta["neutral"] = int(m_meta.group(3))
                    # 去掉尾部计数得到纯正文
                    content = re.sub(
                        r"\(helpful=.*\)\s*$",
                        "",
                        content_full,
                        flags=re.IGNORECASE,
                    ).strip()
                else:
                    # 没有计数就直接清理常见样式
                    content = content_full.split("(helpful")[0].strip()

                pb.add_bullet(
                    section=current_section,
                    content=content,
                    bullet_id=bullet_id,
                    metadata=meta,
                )

    try:
        max_numeric = 0
        for b in pb.bullets():
            m_num = re.search(r"-(\d+)$", b.id)
            if m_num:
                val = int(m_num.group(1))
                if val > max_numeric:
                    max_numeric = val
        pb._next_id = max(pb._next_id, max_numeric)
    except Exception:
        pass

    print(f"[INFO] Loaded {len(pb.bullets())} bullets from Final Playbook in {path}")
    return pb


@dataclass
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""

    sample_id: str = ""


class FireInvestigationEnvironment(TaskEnvironment):
    """
    环境打分版本：直接从同一个 DataFrame 读取 Top5_Hit。
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.hit_map = self._build_hit_map(df)

    def _build_hit_map(self, df: pd.DataFrame) -> dict:
        hit_map = {}
        if "sample_id" in df.columns:
            for _, row in df.iterrows():
                hit_map[str(row["sample_id"]).strip()] = int(row.get("Top5_Hit", 0))
        else:
            for idx, row in enumerate(df.itertuples(), start=1):
                hit_map[f"q{idx:02d}"] = int(getattr(row, "Top5_Hit", 0))
        return hit_map

    def evaluate(self, sample, generator_output):
        sid = getattr(sample, "sample_id", "")
        top5_hit = self.hit_map.get(sid, 0)
        status = "aligned" if top5_hit == 1 else "divergent"
        feedback = (
            f"Top5_Hit={top5_hit} → {status}. "
            "If divergent, incorporate missing medical details from the true diagnosis."
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={"Top5_Hit": top5_hit},
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel",
        default="/gpfs/flash/home/wyj/futong/output/guilin_8_1_evaluated2.xlsx",
        help="Path to the Excel file that already contains model diagnoses.",
    )
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Model used for Reflector/Curator (or remote model name).",
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        help="LLM backend to use (transformers / openai).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the markdown report. If omitted, a timestamped file under reports/ is used.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="Comma-separated CUDA device ids to expose (default: 2,3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of offline adaptation epochs."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default deterministic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N rows from the Excel file (optional).",
    )
    return parser.parse_args()


def load_questions(df: pd.DataFrame) -> List[QuestionSample]:
    """
    从 Excel 读取病例，并封装成 OfflineAdapter 可以吃的样本。

    Version 3: 统一经验库版本，只读取经验库的 bullet_id。
    - 从 retrieved_experiences_json 列读取经验库的 bullet_id
    """

    def safe_get(row, colname: str) -> str:
        if colname in row and pd.notna(row[colname]):
            return str(row[colname]).strip()
        return ""

    def extract_bullet_ids_from_json(raw_json: str) -> List[str]:
        """从 JSON 字符串中提取 bullet_id 列表"""
        bullet_ids = []
        if not raw_json:
            return bullet_ids

        try:
            items = json.loads(raw_json)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        # 支持多种可能的字段名
                        bid = None
                        for key in ["bullet_id", "id", "bulletId"]:
                            if key in item:
                                bid = str(item[key]).strip()
                                break
                        if bid:
                            bullet_ids.append(bid)
        except Exception as e:
            print(f"[WARN] JSON 解析失败: {e}")
        return bullet_ids

    samples: List[QuestionSample] = []

    for idx, row in df.iterrows():
        parts = []

        # 1) 性别
        sex = safe_get(row, "性别_clean")
        if sex:
            parts.append(f"[性别] {sex}")

        # 2) 年龄
        age = safe_get(row, "年龄_clean")
        if age:
            parts.append(f"[年龄] {age}")

        # 3) 病历
        note = safe_get(row, "病历_clean")
        if note:
            parts.append(f"[病历] {note}")

        # 4) 检验
        labs = safe_get(row, "检验结果")
        if labs:
            parts.append(f"[检验结果] {labs}")

        # 5) 检查
        exams = safe_get(row, "检查结果")
        if exams:
            parts.append(f"[检查结果] {exams}")

        question_text = "\n".join(parts).strip()

        # ground truth
        ground_truth = safe_get(row, "诊断") or None

        # model diagnosis info
        most_likely = safe_get(row, "most_likely_diagnosis")
        rationale   = safe_get(row, "diagnostic_rationale")

        # -----------------------------
        # ⭐ Version 3: 只读取经验库的 bullet_id
        # -----------------------------
        # 读取经验库的 bullet_id
        exp_json = safe_get(row, "retrieved_experiences_json")
        bullet_ids = extract_bullet_ids_from_json(exp_json)

        # 兼容旧版本：如果新列不存在，尝试读取旧列名
        if not bullet_ids:
            old_json = safe_get(row, "retrieved_memories_json")
            if old_json:
                bullet_ids = extract_bullet_ids_from_json(old_json)
                print(f"[INFO] Row {idx}: 使用旧列名 retrieved_memories_json")

        if bullet_ids:
            print(f"[DEBUG] Row {idx}: 经验库 {len(bullet_ids)} 个 bullet_id")

        samples.append(
            QuestionSample(
                sample_id=f"q{len(samples)+1:02d}",
                question=question_text,
                context="结构化后的病历/性别/年龄/检查检验，请聚焦异常项。",
                ground_truth=ground_truth,
                metadata={
                    "most_likely_diagnosis": most_likely,
                    "diagnostic_rationale": rationale,
                    "bullet_ids": bullet_ids,  # ⭐经验库的 bullet_id 列表
                },
            )
        )

    return samples



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_results(results: Iterable[AdapterStepResult]) -> Dict[str, float]:
    scores = [
        step.environment_result.metrics.get("Top5_Hit", 0.0)
        for step in results
    ]
    if not scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": mean(scores), "min": min(scores), "max": max(scores)}


def truncate(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def build_report(
    args: argparse.Namespace,
    results: List[AdapterStepResult],
    playbook_text: str,        # 诊断策略 playbook
) -> str:
    stats = summarize_results(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# Questions Test Report")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- CUDA devices: `{args.cuda_visible_devices}`")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Samples: {len(results)}")
    lines.append(
        f"- Top5匹配率 (avg/min/max): {stats['avg']:.2%} / "
        f"{stats['min']:.2%} / {stats['max']:.2%}"
    )
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | Top5_Hit | Question | Final Answer (truncated) |")
    lines.append("|---|------------|----------|--------------------------|")
    for step in results:
        score = step.environment_result.metrics.get("Top5_Hit", 0.0)
        question = truncate(step.sample.question)
        final_answer = truncate(step.generator_output.final_answer or "")
        lines.append(
            f"| {step.sample.sample_id} | {score:.2%} | {question} | {final_answer} |"
        )
    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")
    for step in results:
        score = step.environment_result.metrics.get("Top5_Hit", 0.0)
        lines.append(f"### {step.sample.sample_id} — Similarity {score:.2%}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(step.sample.question)
        lines.append("")
        lines.append("**Model Final Answer**")
        lines.append("")
        lines.append(step.generator_output.final_answer or "(empty)")
        lines.append("")
        lines.append("**Retrieved Bullet IDs**")
        lines.append("")
        lines.append(json.dumps(step.generator_output.bullet_ids, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("**Reference Answer**")
        lines.append("")
        lines.append(step.environment_result.ground_truth or "(none)")
        lines.append("")
        lines.append("**Environment Feedback**")
        lines.append("")
        lines.append(step.environment_result.feedback)
        lines.append("")
        lines.append("**Reflection Snapshot**")
        lines.append("")
        lines.append(json.dumps(step.reflection.raw, ensure_ascii=False, indent=2))
        lines.append("")
        if step.reflection.bullet_tags:
            lines.append("**Tag Reasons**")
            lines.append("")
            for bt in step.reflection.bullet_tags:
                lines.append(f"- [{bt.id}] {bt.tag}: {bt.reason}")
            lines.append("")
        lines.append("**Curator Operations**")
        lines.append("")
        lines.append(json.dumps(step.curator_output.raw, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("**Playbook Excerpt (from retrieved bullet_ids)**")
        lines.append("")
        lines.append(step.reflection.raw.get("_playbook_excerpt", ""))
        lines.append("")

    # 使用字符串形式的主 playbook（诊断策略）
    lines.append("## Final Playbook")
    lines.append("")
    lines.append(playbook_text or "(playbook is empty)")
    lines.append("")

    return "\n".join(lines)



def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    excel_path = Path(args.excel)
    df = pd.read_excel(excel_path)
    # If a row limit is specified, slice the dataframe before building samples
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit]
        print(f"Using only the first {args.limit} rows from {excel_path}.")
    samples = load_questions(df)
    if samples:
        print("[DEBUG] First sample metadata:", samples[0].metadata)

    print(f"Loaded {len(samples)} questions from {excel_path}.")
    print(
        f"Loading model for Reflector/Curator from {args.model_path} "
        f"on GPUs {args.cuda_visible_devices}..."
    )

    # 这个 client 给 Reflector / Curator 用
    client = UniversalLLMClient(
        args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    # 你自己的生成器逻辑
    generator = Generator(llm=None)
    reflector = Reflector(client)
    curator = Curator(client)

    # 一份"全局"的 playbook，大家都往这里写
    PLAYBOOK_REPORT = "/gpfs/flash/home/wyj/futong/ACE-open-main/reports/questions_report_20251204_034939.md"
    global_playbook = load_playbook_from_report(PLAYBOOK_REPORT)

    # 一份"全局"的 adapter，用来维护近期反思窗口等状态
    global_adapter = OfflineAdapter(
        playbook=global_playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=3,
    )

    # 一把全局锁，写 playbook 时用
    playbook_lock = threading.Lock()

    environment = FireInvestigationEnvironment(df)

    # ===== 下面是并行化 + 进度条部分 =====
    max_workers = 128

    def run_one_sample(sample):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # 0. 在锁里读取一次全局对象，做快照，供后续 LLM 调用使用
                with playbook_lock:
                    reflection_ctx_snapshot = global_adapter._reflection_context()

                # 1. 先并行做"不会改全局状态"的部分（使用快照，不再碰全局）
                generator_output = generator.generate(
                    question=sample.question,
                    context=sample.context,
                    playbook=global_playbook,
                    reflection=reflection_ctx_snapshot,

                    # 把 Excel 里的 most_likely_diagnosis / diagnostic_rationale / bullet_ids 全部传进去
                    **(sample.metadata or {}),

                    # 如果你想保留单独的 retrieved_bullet_ids，也可以一起传
                    retrieved_bullet_ids=sample.metadata.get("bullet_ids", []),
                )


                env_result = environment.evaluate(sample, generator_output)

                # 所有样本都进入统一的ACE反思流程
                # 基于按需查询构造 playbook 摘要
                excerpt_lines = []
                _seen = set()

                for bid in generator_output.bullet_ids:
                    if bid in _seen:
                        continue

                    # 从经验库查询内容
                    content = global_playbook.get_bullet_content(bid)

                    if content:
                        excerpt_lines.append(f"[{bid}] {content}")
                        _seen.add(bid)

                playbook_excerpt = "\n".join(excerpt_lines)

                reflection = reflector.reflect(
                    question=sample.question,
                    generator_output=generator_output,
                    playbook=global_playbook,
                    ground_truth=env_result.ground_truth,
                    feedback=env_result.feedback,
                    max_refinement_rounds=global_adapter.max_refinement_rounds,
                    playbook_excerpt=playbook_excerpt,
                    allowed_ids=", ".join(generator_output.bullet_ids),
                )
                try:
                    reflection.raw["_playbook_excerpt"] = playbook_excerpt
                except Exception:
                    pass

                # 在加锁前就能算出来的东西，尽量放在锁外
                question_ctx = global_adapter._question_context(sample, env_result)
                progress_str = global_adapter._progress_string(1, 1, 1, 1)  # 简单给个占位

                curator_output = curator.curate(
                    reflection=reflection,
                    playbook=global_playbook,
                    question_context=question_ctx,
                    progress=progress_str,
                    playbook_text="",
                )

                curator_output.delta.operations = [
                    op for op in curator_output.delta.operations if str(getattr(op, "type", "")).upper() != "TAG"
                ]
                try:
                    raw_ops = curator_output.raw.get("operations")
                    if isinstance(raw_ops, list):
                        curator_output.raw["operations"] = [
                            item for item in raw_ops
                            if not (isinstance(item, dict) and str(item.get("type", "")).upper() == "TAG")
                        ]
                except Exception:
                    pass

                # 修改全局状态时加锁
                with playbook_lock:
                    if generator_output.bullet_ids:
                        global_adapter._apply_bullet_tags(reflection)

                    global_adapter._update_recent_reflections(reflection)
                    global_playbook.apply_delta(curator_output.delta)
                snapshot = ""

                # 3. 返回这条样本的结果
                return AdapterStepResult(
                    sample=sample,
                    generator_output=generator_output,
                    environment_result=env_result,
                    reflection=reflection,
                    curator_output=curator_output,
                    playbook_snapshot=snapshot,
                ), None  # 第二个返回值现在可以不用 playbook 了

            except Exception as e:
                sample_id = getattr(sample, "sample_id", "?")
                print(f"[retry] sample {sample_id} attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(1.0)



    results = []

    print("Starting offline adaptation in parallel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_sample, s) for s in samples]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Adapting"):
            try:
                step, _ = fut.result()   # 第二个返回值现在不用了
                results.append(step)
            except Exception as e:
                print(f"[WARN] Sample failed: {repr(e)}")
                continue

    # 所有线程都跑完后，global_playbook 就是最终版本
    combined_playbook_text = global_playbook.as_prompt() or "(playbook is empty)"

    # 报告中写入诊断策略 playbook
    report_markdown = build_report(
        args,
        results,
        combined_playbook_text,
    )
    # Create a timestamped output path to avoid overwriting
    timestamp_local = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
        # If target exists, suffix with timestamp to avoid overwriting
        if output_path.exists():
            output_path = output_path.with_name(
                f"{output_path.stem}_{timestamp_local}{output_path.suffix}"
            )
    else:
        output_path = ROOT / "reports" / f"questions_report_{timestamp_local}.md"

    ensure_parent(output_path)
    output_path.write_text(report_markdown, encoding="utf-8")
    print(f"Report written to {output_path}")

if __name__ == "__main__":
    main()

