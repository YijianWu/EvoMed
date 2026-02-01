#!/usr/bin/env python3
"""Run Engine modular evolution with fixed/mutable module separation.

Modular Evolution:
- Fixed Module (contextual_states + decision_behaviors): for vector retrieval, immutable
- Iterative Module (uncertainty + delayed_assumptions): mutable in reflection phase

Process:
1. First batch: construct modular experience library
2. Subsequent batches: retrieval based on fixed module → reflection evaluation + iterative module update → Curator (ADD/UPDATE_MUTABLE/TAG)
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

from evomed.evomem.engine import (
    # Traditional components
    AdapterStepResult,
    Curator,
    EnvironmentResult,
    GeneratorOutput,
    OfflineAdapter,
    Reflector,
    Sample,
    TaskEnvironment,
    UniversalLLMClient,
    # Modular components
    ModularPlaybook,
    ModularBullet,
    ModularSemanticRetriever,
    ModularRetrievalResult,
    ModularReflector,
    ModularReflectorOutput,
    build_modular_excerpt,
)
from evomed.evomem.engine.delta import DeltaBatch, DeltaOperation

# Import prompts
from evomed.evomem.engine.prompts import CURATOR_PROMPT
from evomed.evomem.engine.roles import MODULAR_REFLECTOR_PROMPT


@dataclass
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""
    sample_id: str = ""


class ModularEnvironment(TaskEnvironment):
    """Modular evolution environment"""

    def evaluate(self, sample, generator_output):
        # Use gold standard diagnosis
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
        default="/gpfs/flash/home/wyj/futong/output/guilin_100K_20K_diagnosis_20260102_161948.xlsx",
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
    """Load batch samples"""

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

        sex = safe_get(row, "Gender_clean")
        if sex:
            parts.append(f"[Gender] {sex}")

        age = safe_get(row, "Age_clean")
        if age:
            parts.append(f"[Age] {age}")

        note = safe_get(row, "Record_clean")
        if note:
            parts.append(f"[Record] {note}")

        labs = safe_get(row, "Labs")
        if labs:
            parts.append(f"[Labs] {labs}")

        exams = safe_get(row, "Exams")
        if exams:
            parts.append(f"[Exams] {exams}")

        question_text = "\n".join(parts).strip()
        ground_truth = safe_get(row, "Diagnosis") or None

        # Use gold standard as diagnosis result
        most_likely = ground_truth or "Unknown Diagnosis"
        rationale = f"Based on gold standard diagnosis: {ground_truth}"

        samples.append(
            QuestionSample(
                sample_id=f"q{len(samples)+1:02d}",
                question=question_text,
                context="Structured medical record data",
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
    modular_reflector: ModularReflector,
    curator: Curator,
    environment: ModularEnvironment,
    batch_id: int,
    modular_retriever: Optional[ModularSemanticRetriever],
    modular_playbook: ModularPlaybook,
    retrieval_top_k: int,
    duplicate_threshold: float,
    similarity_threshold: float,  # New: similarity threshold to decide update or add
    playbook_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    Process modular evolution for a single sample
    
    Core Logic:
    - First batch: direct ADD operation for new experience
    - Subsequent batches:
      - similarity >= similarity_threshold: modify iterative module (UPDATE_MUTABLE)
      - similarity < similarity_threshold: add new experience (ADD)
    """
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # 1. Generate diagnosis (Directly from metadata)
            metadata = sample.metadata or {}
            most_likely = str(metadata.get("most_likely_diagnosis", ""))
            rationale = str(metadata.get("diagnostic_rationale", ""))
            bullet_ids = metadata.get("bullet_ids") or metadata.get("retrieved_bullet_ids") or []
            if not isinstance(bullet_ids, list):
                bullet_ids = []

            generator_output = GeneratorOutput(
                reasoning=rationale,
                final_answer=most_likely,
                bullet_ids=bullet_ids,
                raw={
                    "most_likely_diagnosis": most_likely,
                    "diagnostic_rationale": rationale,
                    "question": sample.question,
                    "context": sample.context,
                },
            )

            # 2. Environment assessment
            env_result = environment.evaluate(sample, generator_output)

            # 3. Modular retrieval and reflection
            retrieved_results: List[ModularRetrievalResult] = []
            high_similarity_results: List[ModularRetrievalResult] = []  # High similarity results
            mutable_updates = []
            bullet_tags = []
            should_add_new = True  # Whether to add new experience

            if batch_id == 1:
                # First batch: no retrieval, generate new experience from reflection directly
                modular_excerpts = "(first batch - no existing experiences)"
                should_add_new = True
            else:
                # Subsequent batches: retrieval based on fixed module
                if modular_retriever is not None:
                    query = env_result.ground_truth or sample.question
                    retrieved_results = modular_retriever.search_similar(
                        query, top_k=retrieval_top_k
                    )
                    
                    # Filter high similarity results
                    high_similarity_results = [
                        r for r in retrieved_results if r.score >= similarity_threshold
                    ]
                    
                    print(f"[RETRIEVAL] Found {len(retrieved_results)} results, "
                          f"{len(high_similarity_results)} above threshold ({similarity_threshold})")
                    
                    if high_similarity_results:
                        # High similarity match: trigger UPDATE_MUTABLE, don't add new experience
                        should_add_new = False
                        print(f"[DECISION] High similarity found -> UPDATE_MUTABLE mode")
                        for r in high_similarity_results[:3]:
                            print(f"  - {r.bullet_id}: score={r.score:.3f}")
                    else:
                        # No high similarity match: need to add new experience
                        should_add_new = True
                        print(f"[DECISION] No high similarity -> ADD new experience mode")

                # Build modular experience summary
                modular_excerpts = build_modular_excerpt(retrieved_results)

            # 4. Modular reflection
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

            # 5. Curator decision
            with playbook_lock:
                playbook_text = modular_playbook.as_prompt()
                playbook_stats = json.dumps(modular_playbook.stats(), ensure_ascii=False)

            question_ctx = f"""
question: {sample.question}
ground_truth: {env_result.ground_truth}
feedback: {env_result.feedback}
"""

            # Convert to traditional ReflectorOutput for Curator compatibility
            compat_reflection = reflection.to_reflector_output()

            curator_output = curator.curate(
                reflection=compat_reflection,
                playbook=modular_playbook,
                question_context=question_ctx,
                progress=f"batch {batch_id}",
                playbook_text=playbook_text,
                playbook_stats=playbook_stats,
            )

            # 6. Collect operation results
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
                "should_add_new": should_add_new,  # Whether to add new experience
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
    Batch apply modular operations
    
    Core Logic:
    - should_add_new=True: execute ADD operation
    - should_add_new=False: execute UPDATE_MUTABLE operation (modify existing experience)
    - TAG operation always executed
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

        # Apply TAG operation (from reflection, always executed)
        if hasattr(reflection, "bullet_tags"):
            for bt in reflection.bullet_tags:
                bullet = modular_playbook.get_modular_bullet(bt.id)
                if bullet:
                    modular_playbook.tag_modular_bullet(bt.id, bt.tag)
                    stats["tag_count"] += 1

        # Apply UPDATE_MUTABLE operation (from reflection)
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
                        # Synchronously update retriever cache
                        modular_retriever.update_mutable_modules(
                            update.bullet_id, mutable_data
                        )
                        stats["update_mutable_count"] += 1
                        print(f"[UPDATE_MUTABLE] Updated {update.bullet_id}")

        # Apply Curator operations
        for op in curator_output.delta.operations:
            op_type = op.type.upper()

            if op_type == "ADD":
                # If high similarity match exists, skip ADD and trigger UPDATE_MUTABLE instead
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

                # Duplication check
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
                    # Add to retriever
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
    """Save modular experience library"""
    output_path.write_text(playbook.dumps(), encoding="utf-8")
    print(f"[INFO] Saved modular playbook to {output_path}")


def load_modular_playbook(path: str) -> ModularPlaybook:
    """Load modular experience library"""
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
    """Build report"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    
    lines = [
        "# Engine Modular Evolution Report",
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

    # Add experience library content
    for section, bullet_ids in sorted(playbook._sections.items()):
        lines.append(f"### {section}")
        lines.append("")
        for bullet_id in bullet_ids:
            bullet = playbook._modular_bullets.get(bullet_id)
            if bullet:
                lines.append(f"**[{bullet.id}]** (helpful={bullet.helpful}, harmful={bullet.harmful})")
                lines.append("")
                lines.append("Fixed Module:")
                lines.append(f"- Scenario: {bullet.fixed_modules.contextual_states.scenario}")
                lines.append(f"- Chief Complaint: {bullet.fixed_modules.contextual_states.chief_complaint}")
                lines.append(f"- Core Symptoms: {bullet.fixed_modules.contextual_states.core_symptoms}")
                lines.append(f"- Diagnostic Path: {bullet.fixed_modules.decision_behaviors.diagnostic_path}")
                lines.append("")
                lines.append("Iterative Module:")
                lines.append(f"- Uncertainty: {bullet.mutable_modules.uncertainty.primary_uncertainty}")
                if bullet.mutable_modules.delayed_assumptions.pending_validations:
                    validations = ", ".join(bullet.mutable_modules.delayed_assumptions.pending_validations)
                    lines.append(f"- To be verified: {validations}")
                lines.append("")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Load data
    df = pd.read_excel(args.excel)
    if args.limit:
        df = df.iloc[:args.limit]

    samples = load_questions_batch(df, args.batch_id, args.batch_size)
    print(f"[INFO] Loaded {len(samples)} samples for batch {args.batch_id}")

    # Initialize LLM
    client = UniversalLLMClient(
        args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    # Initialize components
    modular_reflector = ModularReflector(client, MODULAR_REFLECTOR_PROMPT)
    curator = Curator(client, CURATOR_PROMPT)
    environment = ModularEnvironment()

    # Initialize modular experience library and retriever
    if args.batch_id == 1:
        modular_playbook = ModularPlaybook()
        modular_retriever = ModularSemanticRetriever()
    else:
        if args.previous_playbook:
            modular_playbook = load_modular_playbook(args.previous_playbook)
        else:
            # Auto find
            reports_dir = ROOT / "outputs" / "reports"
            playbook_files = list(reports_dir.glob(f"modular_playbook_batch{args.batch_id-1}_*.json"))
            if playbook_files:
                latest = max(playbook_files, key=lambda p: p.stat().st_mtime)
                modular_playbook = load_modular_playbook(str(latest))
            else:
                print("[ERROR] No previous playbook found")
                sys.exit(1)

        # Initialize retriever and index existing experiences
        modular_retriever = ModularSemanticRetriever()
        for bullet_id, bullet in modular_playbook._modular_bullets.items():
            modular_retriever.add_modular_experience(
                bullet_id=bullet_id,
                section=bullet.section,
                fixed_modules=bullet.fixed_modules.to_dict(),
                mutable_modules=bullet.mutable_modules.to_dict(),
            )
        print(f"[INFO] Indexed {len(modular_playbook._modular_bullets)} experiences")

    # Parallel processing
    playbook_lock = threading.Lock()
    results: List[Dict[str, Any]] = []

    print(f"[INFO] Starting modular evolution for batch {args.batch_id}...")

    max_workers = min(64, len(samples))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_modular_sample,
                sample,
                modular_reflector,
                curator,
                environment,
                args.batch_id,
                modular_retriever if args.batch_id > 1 else None,
                modular_playbook,
                args.retrieval_top_k,
                args.duplicate_threshold,
                args.similarity_threshold,  # Similarity threshold
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

    # Batch apply operations
    print("[INFO] Applying modular operations...")
    with playbook_lock:
        stats = apply_modular_operations(
            modular_playbook,
            modular_retriever,
            results,
            args.duplicate_threshold,
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save modular experience library
    playbook_path = reports_dir / f"modular_playbook_batch{args.batch_id}_{timestamp}.json"
    save_modular_playbook(modular_playbook, playbook_path)

    # Generate report
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
