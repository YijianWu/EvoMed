#!/usr/bin/env python3
"""Run ACE modular evolution with fixed/mutable module separation.

Modular Evolution:
- 固定模块（contextual_states + decision_behaviors）：用于向量检索，不可修改
- 可迭代模块（uncertainty + delayed_assumptions）：反思阶段可修改

流程：
1. 第一批次：构建模块化经验库
2. 后续批次：基于固定模块检索 → 反思评估 + 更新可迭代模块 → Curator(ADD/UPDATE_MUTABLE/TAG)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
    # 传统组件
    AdapterStepResult,
    Curator,
    EnvironmentResult,
    Generator,
    OfflineAdapter,
    Reflector,
    Sample,
    TaskEnvironment,
    UniversalLLMClient,
    # 模块化组件
    ModularPlaybook,
    ModularBullet,
    ModularSemanticRetriever,
    ModularRetrievalResult,
    ModularReflector,
    ModularReflectorOutput,
    build_modular_excerpt,
)
from ace.delta import DeltaBatch, DeltaOperation

# 导入 prompts
from prompts import CURATOR_PROMPT
from ace.roles import MODULAR_REFLECTOR_PROMPT


@dataclass
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""
    sample_id: str = ""


class ModularEnvironment(TaskEnvironment):
    """模块化演化环境"""

    def evaluate(self, sample, generator_output):
        # 使用金标准诊断
        top5_hit = 1
        status = "aligned"
        feedback = (
            f"Top5_Hit={top5_hit} → {status}. "
            "Using ground truth as diagnosis result for modular evolution."
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
        default="/gpfs/flash/home/wyj/futong/output/guilin_100K_20K_诊断_20260102_161948.xlsx",
        help="Path to the Excel file that contains patient data.",
    )
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Model used for Reflector/Curator.",
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        help="LLM backend to use (transformers / openai).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the report.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="CUDA device ids.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of offline adaptation epochs."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int, default=1024,
        help="Maximum tokens per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float, default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--limit",
        type=int, default=None,
        help="Only use first N rows.",
    )
    parser.add_argument(
        "--batch-id",
        type=int, required=True,
        help="Batch ID to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int, default=1000,
        help="Samples per batch.",
    )
    parser.add_argument(
        "--previous-playbook",
        default=None,
        help="Path to previous modular playbook JSON.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int, default=5,
        help="Number of top retrieval results.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float, default=0.95,
        help="Similarity threshold for duplicate detection (higher = less strict).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float, default=0.80,
        help="Similarity threshold: >= this triggers UPDATE_MUTABLE, < this triggers ADD.",
    )
    return parser.parse_args()


def load_questions_batch(df: pd.DataFrame, batch_id: int, batch_size: int) -> List[QuestionSample]:
    """加载批次样本"""

    def safe_get(row, colname: str) -> str:
        try:
            value = getattr(row, colname, None)
            if value is not None and pd.notna(value):
                return str(value).strip()
        except:
            pass
        return ""

    start_idx = (batch_id - 1) * batch_size
    end_idx = min(batch_id * batch_size, len(df))

    print(f"[INFO] Batch {batch_id}: processing samples {start_idx} to {end_idx-1}")

    batch_df = df.iloc[start_idx:end_idx]
    samples: List[QuestionSample] = []

    for idx, row in enumerate(batch_df.itertuples(), start=start_idx):
        parts = []

        sex = safe_get(row, "性别_clean")
        if sex:
            parts.append(f"[性别] {sex}")

        age = safe_get(row, "年龄_clean")
        if age:
            parts.append(f"[年龄] {age}")

        note = safe_get(row, "病历_clean")
        if note:
            parts.append(f"[病历] {note}")

        labs = safe_get(row, "检验结果")
        if labs:
            parts.append(f"[检验结果] {labs}")

        exams = safe_get(row, "检查结果")
        if exams:
            parts.append(f"[检查结果] {exams}")

        question_text = "\n".join(parts).strip()
        ground_truth = safe_get(row, "诊断") or None

        # 使用金标准作为诊断结果
        most_likely = ground_truth or "未知诊断"
        rationale = f"基于金标准诊断：{ground_truth}"

        samples.append(
            QuestionSample(
                sample_id=f"q{len(samples)+1:02d}",
                question=question_text,
                context="结构化病历数据",
                ground_truth=ground_truth,
                metadata={
                    "most_likely_diagnosis": most_likely,
                    "diagnostic_rationale": rationale,
                    "batch_id": batch_id,
                },
            )
        )

    return samples


def process_modular_sample(
    sample: QuestionSample,
    generator: Generator,
    modular_reflector: ModularReflector,
    curator: Curator,
    environment: ModularEnvironment,
    batch_id: int,
    modular_retriever: Optional[ModularSemanticRetriever],
    modular_playbook: ModularPlaybook,
    retrieval_top_k: int,
    duplicate_threshold: float,
    similarity_threshold: float,  # 新增：相似度阈值，决定是更新还是添加
    playbook_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    处理单个样本的模块化演化
    
    核心逻辑：
    - 第一批次：直接添加新经验（ADD）
    - 后续批次：
      - 相似度 >= similarity_threshold：修改可迭代模块（UPDATE_MUTABLE）
      - 相似度 < similarity_threshold：添加新经验（ADD）
    """
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # 1. 生成诊断
            generator_output = generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=modular_playbook,  # 兼容性参数
                reflection="",
                **(sample.metadata or {})
            )

            # 2. 环境评估
            env_result = environment.evaluate(sample, generator_output)

            # 3. 模块化检索和反思
            retrieved_results: List[ModularRetrievalResult] = []
            high_similarity_results: List[ModularRetrievalResult] = []  # 高相似度结果
            mutable_updates = []
            bullet_tags = []
            should_add_new = True  # 是否需要添加新经验

            if batch_id == 1:
                # 第一批次：无检索，直接反思生成新经验
                modular_excerpts = "(first batch - no existing experiences)"
                should_add_new = True
            else:
                # 后续批次：基于固定模块检索
                if modular_retriever is not None:
                    query = env_result.ground_truth or sample.question
                    retrieved_results = modular_retriever.search_similar(
                        query, top_k=retrieval_top_k
                    )
                    
                    # 筛选高相似度结果
                    high_similarity_results = [
                        r for r in retrieved_results if r.score >= similarity_threshold
                    ]
                    
                    print(f"[RETRIEVAL] Found {len(retrieved_results)} results, "
                          f"{len(high_similarity_results)} above threshold ({similarity_threshold})")
                    
                    if high_similarity_results:
                        # 有高相似度匹配：触发 UPDATE_MUTABLE，不添加新经验
                        should_add_new = False
                        print(f"[DECISION] High similarity found -> UPDATE_MUTABLE mode")
                        for r in high_similarity_results[:3]:
                            print(f"  - {r.bullet_id}: score={r.score:.3f}")
                    else:
                        # 无高相似度匹配：需要添加新经验
                        should_add_new = True
                        print(f"[DECISION] No high similarity -> ADD new experience mode")

                # 构建模块化经验摘要
                modular_excerpts = build_modular_excerpt(retrieved_results)

            # 4. 模块化反思
            allowed_ids = [r.bullet_id for r in retrieved_results] if retrieved_results else None
            
            reflection = modular_reflector.reflect(
                question=sample.question,
                generator_output=generator_output,
                modular_excerpts=modular_excerpts,
                ground_truth=env_result.ground_truth,
                feedback=env_result.feedback,
                max_refinement_rounds=1,
                allowed_ids=allowed_ids,
            )

            bullet_tags = reflection.bullet_tags
            mutable_updates = reflection.mutable_updates

            # 5. Curator 决策
            with playbook_lock:
                playbook_text = modular_playbook.as_prompt()
                playbook_stats = json.dumps(modular_playbook.stats(), ensure_ascii=False)

            question_ctx = f"""
question: {sample.question}
ground_truth: {env_result.ground_truth}
feedback: {env_result.feedback}
"""

            # 转换为传统 ReflectorOutput 以兼容 Curator
            compat_reflection = reflection.to_reflector_output()

            curator_output = curator.curate(
                reflection=compat_reflection,
                playbook=modular_playbook,
                question_context=question_ctx,
                progress=f"batch {batch_id}",
                playbook_text=playbook_text,
                playbook_stats=playbook_stats,
            )

            # 6. 收集操作结果
            result_data = {
                "sample": sample,
                "generator_output": generator_output,
                "env_result": env_result,
                "reflection": reflection,
                "curator_output": curator_output,
                "retrieved_results": retrieved_results,
                "high_similarity_results": high_similarity_results,
                "mutable_updates": mutable_updates,
                "bullet_tags": bullet_tags,
                "should_add_new": should_add_new,  # 是否需要添加新经验
            }

            return result_data

        except Exception as e:
            print(f"[RETRY] Sample {sample.sample_id} attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(1.0)


def apply_modular_operations(
    modular_playbook: ModularPlaybook,
    modular_retriever: ModularSemanticRetriever,
    results: List[Dict[str, Any]],
    duplicate_threshold: float,
) -> Dict[str, int]:
    """
    批量应用模块化操作
    
    核心逻辑：
    - should_add_new=True: 执行 ADD 操作
    - should_add_new=False: 执行 UPDATE_MUTABLE 操作（修改现有经验）
    - TAG 操作总是执行
    """
    
    stats = {
        "add_count": 0,
        "update_mutable_count": 0,
        "tag_count": 0,
        "duplicate_filtered": 0,
        "skipped_add_due_to_similarity": 0,
    }

    for result_data in results:
        reflection = result_data["reflection"]
        curator_output = result_data["curator_output"]
        should_add_new = result_data.get("should_add_new", True)
        high_similarity_results = result_data.get("high_similarity_results", [])

        # 应用 TAG 操作（来自反思，总是执行）
        if hasattr(reflection, "bullet_tags"):
            for bt in reflection.bullet_tags:
                bullet = modular_playbook.get_modular_bullet(bt.id)
                if bullet:
                    modular_playbook.tag_modular_bullet(bt.id, bt.tag)
                    stats["tag_count"] += 1

        # 应用 UPDATE_MUTABLE 操作（来自反思）
        if hasattr(reflection, "mutable_updates") and reflection.mutable_updates:
            for update in reflection.mutable_updates:
                mutable_data = {}
                if update.uncertainty:
                    mutable_data["uncertainty"] = update.uncertainty
                if update.delayed_assumptions:
                    mutable_data["delayed_assumptions"] = update.delayed_assumptions
                
                if mutable_data:
                    bullet = modular_playbook.update_mutable_modules(
                        update.bullet_id, mutable_data
                    )
                    if bullet:
                        # 同步更新检索器缓存
                        modular_retriever.update_mutable_modules(
                            update.bullet_id, mutable_data
                        )
                        stats["update_mutable_count"] += 1
                        print(f"[UPDATE_MUTABLE] Updated {update.bullet_id}")

        # 应用 Curator 操作
        for op in curator_output.delta.operations:
            op_type = op.type.upper()

            if op_type == "ADD":
                # 如果有高相似度匹配，跳过 ADD，改为触发 UPDATE_MUTABLE
                if not should_add_new:
                    print(f"[SKIP_ADD] High similarity exists, skipping ADD for this sample")
                    stats["skipped_add_due_to_similarity"] += 1
                    continue
                
                modules = getattr(op, "modules", None) or {}
                fixed_modules = {
                    "contextual_states": modules.get("contextual_states", {}),
                    "decision_behaviors": modules.get("decision_behaviors", {}),
                }
                mutable_modules = {
                    "uncertainty": modules.get("uncertainty", {}),
                    "delayed_assumptions": modules.get("delayed_assumptions", {}),
                }

                # 重复检查
                is_duplicate, similar_id = modular_retriever.check_duplicate(
                    fixed_modules, duplicate_threshold
                )

                if is_duplicate:
                    print(f"[DUPLICATE] Filtered ADD (similar to {similar_id})")
                    stats["duplicate_filtered"] += 1
                else:
                    bullet = modular_playbook.add_modular_bullet(
                        section=op.section,
                        fixed_modules=fixed_modules,
                        mutable_modules=mutable_modules,
                        bullet_id=op.bullet_id,
                        metadata=op.metadata,
                    )
                    # 添加到检索器
                    modular_retriever.add_modular_experience(
                        bullet_id=bullet.id,
                        section=op.section,
                        fixed_modules=fixed_modules,
                        mutable_modules=mutable_modules,
                    )
                    stats["add_count"] += 1
                    print(f"[ADD] New modular experience: {bullet.id}")

            elif op_type == "UPDATE_MUTABLE":
                mutable_data = getattr(op, "mutable_modules", None) or {}
                if op.bullet_id and mutable_data:
                    modular_playbook.update_mutable_modules(op.bullet_id, mutable_data)
                    modular_retriever.update_mutable_modules(op.bullet_id, mutable_data)
                    stats["update_mutable_count"] += 1

            elif op_type == "TAG":
                if op.bullet_id:
                    for tag, increment in op.metadata.items():
                        if tag in ("helpful", "harmful", "neutral"):
                            modular_playbook.tag_modular_bullet(op.bullet_id, tag, int(increment))
                            stats["tag_count"] += 1

    return stats


def save_modular_playbook(playbook: ModularPlaybook, output_path: Path) -> None:
    """保存模块化经验库"""
    output_path.write_text(playbook.dumps(), encoding="utf-8")
    print(f"[INFO] Saved modular playbook to {output_path}")


def load_modular_playbook(path: str) -> ModularPlaybook:
    """加载模块化经验库"""
    if not os.path.exists(path):
        print(f"[WARN] Playbook file not found: {path}")
        return ModularPlaybook()
    
    with open(path, encoding="utf-8") as f:
        content = f.read()
    
    playbook = ModularPlaybook.loads(content)
    print(f"[INFO] Loaded {len(playbook._modular_bullets)} modular bullets")
    return playbook


def build_report(
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    playbook: ModularPlaybook,
    stats: Dict[str, int],
) -> str:
    """构建报告"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    
    lines = [
        "# ACE Modular Evolution Report",
        "",
        f"- Generated: {timestamp}",
        f"- Batch ID: {args.batch_id}",
        f"- Batch Size: {args.batch_size}",
        f"- Samples Processed: {len(results)}",
        f"- Model: `{args.model_path}`",
        "",
        "## Statistics",
        "",
        f"- ADD Operations: {stats['add_count']}",
        f"- UPDATE_MUTABLE Operations: {stats['update_mutable_count']}",
        f"- TAG Operations: {stats['tag_count']}",
        f"- Duplicates Filtered: {stats['duplicate_filtered']}",
        f"- Final Modular Bullets: {len(playbook._modular_bullets)}",
        "",
        "## Modular Experience Library",
        "",
    ]

    # 添加经验库内容
    for section, bullet_ids in sorted(playbook._sections.items()):
        lines.append(f"### {section}")
        lines.append("")
        for bullet_id in bullet_ids:
            bullet = playbook._modular_bullets.get(bullet_id)
            if bullet:
                lines.append(f"**[{bullet.id}]** (helpful={bullet.helpful}, harmful={bullet.harmful})")
                lines.append("")
                lines.append("固定模块：")
                lines.append(f"- 情境: {bullet.fixed_modules.contextual_states.scenario}")
                lines.append(f"- 主诉: {bullet.fixed_modules.contextual_states.chief_complaint}")
                lines.append(f"- 核心症状: {bullet.fixed_modules.contextual_states.core_symptoms}")
                lines.append(f"- 诊断路径: {bullet.fixed_modules.decision_behaviors.diagnostic_path}")
                lines.append("")
                lines.append("可迭代模块：")
                lines.append(f"- 不确定性: {bullet.mutable_modules.uncertainty.primary_uncertainty}")
                if bullet.mutable_modules.delayed_assumptions.pending_validations:
                    validations = ", ".join(bullet.mutable_modules.delayed_assumptions.pending_validations)
                    lines.append(f"- 待验证: {validations}")
                lines.append("")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 加载数据
    df = pd.read_excel(args.excel)
    if args.limit:
        df = df.iloc[:args.limit]

    samples = load_questions_batch(df, args.batch_id, args.batch_size)
    print(f"[INFO] Loaded {len(samples)} samples for batch {args.batch_id}")

    # 初始化 LLM
    client = UniversalLLMClient(
        args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    # 初始化组件
    generator = Generator(llm=None)
    modular_reflector = ModularReflector(client, MODULAR_REFLECTOR_PROMPT)
    curator = Curator(client, CURATOR_PROMPT)
    environment = ModularEnvironment()

    # 初始化模块化经验库和检索器
    if args.batch_id == 1:
        modular_playbook = ModularPlaybook()
        modular_retriever = ModularSemanticRetriever()
    else:
        if args.previous_playbook:
            modular_playbook = load_modular_playbook(args.previous_playbook)
        else:
            # 自动查找
            reports_dir = ROOT / "outputs" / "reports"
            playbook_files = list(reports_dir.glob(f"modular_playbook_batch{args.batch_id-1}_*.json"))
            if playbook_files:
                latest = max(playbook_files, key=lambda p: p.stat().st_mtime)
                modular_playbook = load_modular_playbook(str(latest))
            else:
                print("[ERROR] No previous playbook found")
                sys.exit(1)

        # 初始化检索器并索引已有经验
        modular_retriever = ModularSemanticRetriever()
        for bullet_id, bullet in modular_playbook._modular_bullets.items():
            modular_retriever.add_modular_experience(
                bullet_id=bullet_id,
                section=bullet.section,
                fixed_modules=bullet.fixed_modules.to_dict(),
                mutable_modules=bullet.mutable_modules.to_dict(),
            )
        print(f"[INFO] Indexed {len(modular_playbook._modular_bullets)} experiences")

    # 并行处理
    playbook_lock = threading.Lock()
    results: List[Dict[str, Any]] = []

    print(f"[INFO] Starting modular evolution for batch {args.batch_id}...")

    max_workers = min(64, len(samples))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_modular_sample,
                sample,
                generator,
                modular_reflector,
                curator,
                environment,
                args.batch_id,
                modular_retriever if args.batch_id > 1 else None,
                modular_playbook,
                args.retrieval_top_k,
                args.duplicate_threshold,
                args.similarity_threshold,  # 相似度阈值
                playbook_lock,
            )
            for sample in samples
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = fut.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Sample failed: {e}")

    # 批量应用操作
    print("[INFO] Applying modular operations...")
    with playbook_lock:
        stats = apply_modular_operations(
            modular_playbook,
            modular_retriever,
            results,
            args.duplicate_threshold,
        )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 保存模块化经验库
    playbook_path = reports_dir / f"modular_playbook_batch{args.batch_id}_{timestamp}.json"
    save_modular_playbook(modular_playbook, playbook_path)

    # 生成报告
    report = build_report(args, results, modular_playbook, stats)
    report_path = reports_dir / f"modular_evolution_report_batch{args.batch_id}_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[INFO] Report saved to {report_path}")

    print("\n[DONE] Modular evolution complete!")
    print(f"  - ADD: {stats['add_count']}")
    print(f"  - UPDATE_MUTABLE: {stats['update_mutable_count']}")
    print(f"  - TAG: {stats['tag_count']}")
    print(f"  - Duplicates filtered: {stats['duplicate_filtered']}")
    print(f"  - Skipped ADD (high similarity): {stats.get('skipped_add_due_to_similarity', 0)}")
    print(f"  - Final experience count: {len(modular_playbook._modular_bullets)}")


if __name__ == "__main__":
    main()

