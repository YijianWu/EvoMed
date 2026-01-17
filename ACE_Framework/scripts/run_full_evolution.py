#!/usr/bin/env python3
"""Run full ACE evolution on all samples.

This script runs the ACE adaptation loop over the entire dataset, evolving the experience library (Playbook).
It supports:
- Starting from scratch (empty playbook) or resuming from an existing playbook.
- Internal retrieval of relevant experiences.
- Duplicate detection for new experiences.
- Periodic checkpointing.
- Final export to JSON format (amem compatible).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional
import re

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
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
from ace.retrieval import SemanticRetriever, RetrievalResult


@dataclass
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""
    sample_id: str = ""
    original_idx: int = 0  # To track position in original dataframe


class FireInvestigationEnvironment(TaskEnvironment):
    """
    Environment wrapper.
    In 'gold standard' mode, we assume the ground truth is correct and use it for feedback.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        pass

    def evaluate(self, sample, generator_output):
        # Always return perfect score as we are learning from ground truth
        top5_hit = 1
        status = "aligned"
        feedback = (
            f"Top5_Hit={top5_hit} → {status}. "
            "Using ground truth as diagnosis result for testing ACE reflection mechanism."
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={"Top5_Hit": top5_hit},
        )


def load_playbook_from_file(path: str) -> Playbook:
    """Load Playbook from a file (either markdown report or direct playbook dump)."""
    pb = Playbook()
    current_section = None
    in_playbook = False
    is_direct_playbook = False

    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return pb

    filename = os.path.basename(path)
    # Heuristic to detect file type
    if filename.endswith(".json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Support loading from simple list of bullets or full playbook dict
            if isinstance(data, dict) and "bullets" in data:
                 return Playbook.from_dict(data)
            # TODO: Add more JSON formats if needed
            pass
    
    if filename.startswith("playbook_") and not filename.startswith("report_"):
        is_direct_playbook = True
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if is_direct_playbook:
                in_playbook = True
            else:
                if line.strip() == "## Final Playbook":
                    in_playbook = True
                    continue

            if not in_playbook:
                continue

            # Skip Case Library if present (legacy support, though user asked to clean it)
            if line.strip() == "## Case Library Playbook":
                break

            if line.startswith("## "):
                current_section = line[3:].strip()
                continue

            if not line.strip():
                continue

            # Parse bullet: - [ID] content (helpful=H, harmful=R[, neutral=N])
            m = re.match(r"- \[([^\]]+)\]\s+(.*)", line)
            if m:
                bullet_id = m.group(1).strip()
                content_full = m.group(2).strip()
                meta: Dict[str, int] = {}
                
                # Extract counters
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
                    content = re.sub(
                        r"\(helpful=.*\)\s*$",
                        "",
                        content_full,
                        flags=re.IGNORECASE,
                    ).strip()
                else:
                    content = content_full.split("(helpful")[0].strip()

                pb.add_bullet(
                    section=current_section,
                    content=content,
                    bullet_id=bullet_id,
                    metadata=meta,
                )

    # Update next_id
    try:
        max_numeric = 0
        for b in pb.bullets():
            m_num = re.search(r"-(\d+)$", b.id)
            if m_num:
                val = int(m_num.group(1))
                max_numeric = max(max_numeric, val)
        pb._next_id = max(pb._next_id, max_numeric)
    except Exception:
        pass

    print(f"[INFO] Loaded {len(pb.bullets())} bullets from {path}")
    return pb


def load_all_questions(df: pd.DataFrame) -> List[QuestionSample]:
    """Load all questions from DataFrame."""
    def safe_get(row, colname: str) -> str:
        try:
            val = getattr(row, colname, None)
            if pd.notna(val):
                return str(val).strip()
        except:
            pass
        return ""

    samples = []
    for idx, row in enumerate(df.itertuples(), start=1):
        parts = []
        for field in ["性别_clean", "年龄_clean", "病历_clean", "检验结果", "检查结果"]:
            val = safe_get(row, field)
            if val:
                label = field.replace("_clean", "")
                parts.append(f"[{label}] {val}")
        
        question_text = "\n".join(parts).strip()
        ground_truth = safe_get(row, "诊断")
        
        # Use ground truth as ideal output
        most_likely = ground_truth or "Unknown"
        rationale = f"Based on ground truth: {ground_truth}"

        samples.append(QuestionSample(
            sample_id=f"q{idx:05d}",
            question=question_text,
            context="Structured medical record.",
            ground_truth=ground_truth,
            metadata={
                "most_likely_diagnosis": most_likely,
                "diagnostic_rationale": rationale,
            },
            original_idx=idx
        ))
    return samples


def perform_internal_retrieval(
    retriever: SemanticRetriever,
    query: str,
    top_k: int = 5
) -> List[RetrievalResult]:
    if not query:
        return []
    return retriever.search_similar(query, top_k)


def run_one_sample(
    sample: QuestionSample,
    global_adapter: OfflineAdapter,
    environment: TaskEnvironment,
    retriever: SemanticRetriever,
    retrieval_top_k: int,
    duplicate_threshold: float,
    playbook_lock: threading.Lock,
    playbook_snapshot: Playbook,
    reflection_ctx_snapshot: str,
    is_initial_phase: bool = False
):
    """
    Process a single sample.
    is_initial_phase: If True, we might skip retrieval or treat it differently (e.g. cold start).
    """
    try:
        # 1. Generate (Simulated using metadata)
        # We use the snapshot for reading, but we don't need to lock for generation
        generator_output = global_adapter.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=playbook_snapshot,
            reflection=reflection_ctx_snapshot,
            **(sample.metadata or {})
        )

        env_result = environment.evaluate(sample, generator_output)

        # 2. Retrieval
        retrieved_results = []
        retrieved_bullet_ids = []
        playbook_excerpt = ""

        # Retrieve based on diagnosis (ground truth)
        query = env_result.ground_truth or sample.question
        
        # Use lock only if needed for thread-safety of the retriever (FAISS inside)
        # The retriever implementation handles its own locking for index access
        if not is_initial_phase and retriever.content_cache:
            retrieved_results = perform_internal_retrieval(retriever, query, retrieval_top_k)
            retrieved_bullet_ids = [r.bullet_id for r in retrieved_results]
            
            if retrieved_results:
                lines = ["=== Retrieved Experiences ===", ""]
                for i, res in enumerate(retrieved_results, 1):
                    lines.append(f"[Item {i} - ID:{res.bullet_id}]")
                    lines.append(f"Content: {res.content}")
                    lines.append("---")
                playbook_excerpt = "\n".join(lines)
            else:
                playbook_excerpt = "(no relevant experiences found)"

        # 3. Reflection
        reflection = global_adapter.reflector.reflect(
            question=sample.question,
            generator_output=generator_output,
            playbook=playbook_snapshot,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=global_adapter.max_refinement_rounds,
            playbook_excerpt=playbook_excerpt,
            allowed_ids=retrieved_bullet_ids,
        )

        # 4. Curation
        question_ctx = global_adapter._question_context(sample, env_result)
        progress_str = global_adapter._progress_string(1, 1, 1, 1) # Dummy progress

        curator_output = global_adapter.curator.curate(
            reflection=reflection,
            playbook=playbook_snapshot,
            question_context=question_ctx,
            progress=progress_str,
            playbook_text="" # Snapshot used internally or passed via arg if needed, but curate uses playbook obj
        )

        # Filter operations
        all_ops = curator_output.delta.operations
        add_ops = [op for op in all_ops if str(getattr(op, "type", "")).upper() == "ADD"]
        
        # 调试：记录 Curator 返回的原始操作数
        curator_add_count = len(add_ops)
        
        filtered_add_ops = []
        duplicate_count = 0
        for op in add_ops:
            content = getattr(op, "content", "")
            if content:
                is_dup, similar_id = retriever.check_duplicate(content, duplicate_threshold)
                if not is_dup:
                    filtered_add_ops.append(op)
                else:
                    duplicate_count += 1
        
        # Only keep ADDs and TAGs (TAGs are handled by reflector usually, but curator might pass them through if designed so)
        # In this logic, we only care about new experiences (ADD) and tags.
        # Check if TAGs are in reflection.bullet_tags
        
        final_ops = filtered_add_ops
        # Note: We are returning the delta, not applying it yet.
        
        # Update curator output with filtered ops
        curator_output.delta.operations = final_ops
        
        return {
            'sample_id': sample.sample_id,
            'reflection': reflection,
            'curator_delta': curator_output.delta,
            'retrieved_count': len(retrieved_results),
            'added_count': len(final_ops),
            'curator_add_count': curator_add_count,  # Curator 原始返回的 ADD 数
            'duplicate_filtered': duplicate_count,   # 被重复检测过滤的数量
        }

    except Exception as e:
        print(f"[ERROR] Sample {sample.sample_id} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="ACE Full Evolution Run")
    parser.add_argument("--excel", required=True, help="Path to input Excel file")
    parser.add_argument("--output-dir", default="reports/evolution_full", help="Directory to save results")
    parser.add_argument("--model-path", default="gpt-4o-mini", help="Model path or API model name")
    parser.add_argument("--backend", default="openai", choices=["transformers", "openai"], help="LLM backend to use")
    parser.add_argument("--resume-playbook", help="Path to existing playbook to resume from")
    parser.add_argument("--start-index", type=int, default=0, help="Start processing from this row index")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Save checkpoint every N samples")
    parser.add_argument("--cuda-visible-devices", default="0,1", help="CUDA devices")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--max-workers", type=int, default=128, help="Number of parallel threads")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    ensure_parent = lambda p: Path(p).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"[INFO] Loading data from {args.excel}...")
    df = pd.read_excel(args.excel)
    samples = load_all_questions(df)
    
    if args.start_index > 0:
        samples = samples[args.start_index:]
        print(f"[INFO] Resuming from index {args.start_index}")
        
    if args.limit:
        samples = samples[:args.limit]
        print(f"[INFO] Limiting to {len(samples)} samples")

    print(f"[INFO] Total samples to process: {len(samples)}")

    # Initialize Components
    client = UniversalLLMClient(
        args.model_path,
        backend=args.backend,
        max_new_tokens=512,
        temperature=0.0,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    generator = Generator(llm=None)
    reflector = Reflector(client)
    curator = Curator(client)
    
    # Load or Init Playbook
    if args.resume_playbook:
        print(f"[INFO] Loading playbook from {args.resume_playbook}")
        playbook = load_playbook_from_file(args.resume_playbook)
    else:
        print("[INFO] Starting with empty playbook")
        playbook = Playbook()

    # Initialize Retriever (with lazy loading to avoid blocking on import)
    print("[INFO] Initializing retriever...")
    retriever = SemanticRetriever(lazy_load=True)
    print("[INFO] Indexing playbook into retriever...")
    for bid, bullet in playbook._bullets.items():
        retriever.add_experience(bid, bullet.content)
    
    if playbook._bullets and hasattr(retriever, '_build_faiss_index'):
        print("[INFO] Building FAISS index...")
        retriever._build_faiss_index()
        print(f"[INFO] FAISS index built for {len(playbook._bullets)} experiences")

    global_adapter = OfflineAdapter(
        playbook=playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=2,
    )
    
    playbook_lock = threading.Lock()
    environment = FireInvestigationEnvironment(df)

    # Process Loop
    processed_count = 0
    max_workers = args.max_workers
    print(f"[INFO] Using {max_workers} parallel workers")
    
    # Split into batches for processing
    batch_size = args.batch_size
    total_batches = (len(samples) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        print(f"\n[INFO] Processing Batch {batch_idx + 1}/{total_batches} (Samples {args.start_index + batch_start} - {args.start_index + batch_end})")
        
        # Snapshot for this batch
        with playbook_lock:
            pb_snapshot = playbook # Deep copy if needed, but Python object ref is fine if we don't mutate in place without lock
            # Actually, we should probably rely on the fact that we apply deltas AFTER the batch.
            # So current state is safe to read.
            reflection_ctx = global_adapter._reflection_context()

        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_one_sample, 
                    s, 
                    global_adapter, 
                    environment, 
                    retriever, 
                    args.retrieval_top_k, 
                    args.duplicate_threshold, 
                    playbook_lock, 
                    pb_snapshot, 
                    reflection_ctx,
                    is_initial_phase=(len(playbook._bullets) == 0)
                ) 
                for s in batch_samples
            ]
            
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Batch Progress"):
                res = fut.result()
                if res:
                    batch_results.append(res)
        
        # Apply Deltas
        print(f"[INFO] Applying updates for batch {batch_idx + 1}...")
        added_total = 0
        curator_total_adds = 0
        duplicate_filtered_total = 0
        
        for res in batch_results:
            # 统计调试信息
            curator_total_adds += res.get('curator_add_count', 0)
            duplicate_filtered_total += res.get('duplicate_filtered', 0)
            
            # Apply Tags
            reflection = res['reflection']
            if hasattr(reflection, 'bullet_tags') and reflection.bullet_tags:
                global_adapter._apply_bullet_tags(reflection)
            
            # Apply Adds
            delta = res['curator_delta']
            if delta.operations:
                # Add to playbook
                playbook.apply_delta(delta)
                added_total += res['added_count']
        
        # 打印详细统计
        print(f"[DEBUG] Curator proposed {curator_total_adds} ADD ops, {duplicate_filtered_total} filtered as duplicates, {added_total} will be added")

        # Re-sync retriever
        # Efficient way: only add what's missing
        current_ids = set(playbook._bullets.keys())
        retriever_ids = set(retriever.content_cache.keys())
        new_ids = current_ids - retriever_ids
        for bid in new_ids:
            retriever.add_experience(bid, playbook.get_bullet_content(bid))
        
        if hasattr(retriever, '_build_faiss_index') and new_ids:
             retriever._build_faiss_index()

        print(f"[INFO] Batch finished. Added {len(new_ids)} new experiences. Total: {len(playbook._bullets)}")
        
        # Checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = output_dir / f"playbook_ckpt_{args.start_index + batch_end}_{timestamp}.json"
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(playbook.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")

    # Final Export
    final_path = output_dir / f"final_playbook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(playbook.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[INFO] Run complete. Final playbook saved to {final_path}")
    
    # Export for AMEM (assuming AMEM expects a list of experiences/memories)
    amem_path = output_dir / f"amem_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    amem_data = []
    for b in playbook.bullets():
        amem_data.append({
            "id": b.id,
            "content": b.content,
            "section": b.section,
            "stats": {
                "helpful": b.helpful,
                "harmful": b.harmful,
                "neutral": b.neutral
            },
            "created_at": b.created_at
        })
    with open(amem_path, 'w', encoding='utf-8') as f:
        json.dump(amem_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] AMEM export saved to {amem_path}")


if __name__ == "__main__":
    main()

