#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE-A-mem-sys Integration Script with Evolution (batch-aware)

功能：
- 一次性导入 ACE Experience库（Final Playbook）与Case库（Case Library）到 A-mem-sys
- 支持“首批不演化，后续批次开启演化”，可配置批大小和演化阈值
- 批内处理使用 A-mem-sys 的并行参数（LLM/chroma 多线程），提高吞吐

用法示例：

    cd /gpfs/flash/home/wyj/futong && \
    YUNWU_BASE_URL="https://yunwu.ai/v1" \
    python ace_amem_integration_experience_and_case_evolve.py \
        --api_key "YOUR_API_KEY" \
        --report_path /gpfs/flash/home/wyj/futong/reports/questions_report_guilin10k_limit_10.md \
        --mem_sys_path /gpfs/flash/home/wyj/futong/A-mem-sys \
        --memory_root /gpfs/flash/home/wyj/futong/A-mem-sys/memory_db \
        --experience_collection_name experience \
        --case_collection_name case \
        --batch_size 1000 \
        --evo_threshold 1000

策略：
- 将Experience/Case按 batch_size 切分
- 第 1 批：enable_evolution=False（仅写入，不演化）
- 第 2 批及以后：enable_evolution=True（演化），evo_threshold 默认 1000
  （如果总数 ≤ batch_size，则不会触发演化；需有第二批才会演化）
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ace_amem_integration_auto_tags import ACEExperienceExtractor
from ace_amem_integration_case_library import CaseLibraryExtractor

logger = logging.getLogger(__name__)


def _chunk(items: List[Any], size: int) -> List[List[Any]]:
    """简单切片成批次。"""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_memory_system(
    mem_sys_path: str,
    memory_root: Optional[str],
    collection_name: str,
    api_key: str,
    llm_backend: str,
    llm_model: str,
    evo_threshold: int,
):
    """创建并返回 AgenticMemorySystem，带演化阈值与路径配置。"""
    if not os.path.isdir(mem_sys_path):
        raise FileNotFoundError(f"A-mem-sys path not found: {mem_sys_path}")
    if mem_sys_path not in sys.path:
        sys.path.insert(0, mem_sys_path)

    # collection 级的 chroma 目录
    if memory_root:
        chroma_dir = os.path.join(memory_root, f"chroma_{collection_name}")
    else:
        chroma_dir = os.path.join(mem_sys_path, "memory_db", f"chroma_{collection_name}")
    os.environ["AMEM_CHROMA_PATH"] = chroma_dir
    os.makedirs(chroma_dir, exist_ok=True)

    am_mod = importlib.import_module("agentic_memory.memory_system")
    AgenticMemorySystem = getattr(am_mod, "AgenticMemorySystem")
    memory_system = AgenticMemorySystem(
        llm_backend=llm_backend,
        llm_model=llm_model,
        api_key=api_key,
        evo_threshold=evo_threshold,
    )

    # 兼容不同版本的内部路径字段
    try:
        for attr in ["storage_dir", "db_path", "root_dir", "memory_db_path"]:
            if hasattr(memory_system, attr):
                setattr(memory_system, attr, memory_root)
                break
    except Exception:
        pass

    return memory_system


def _store_batch(
    memory_system,
    contents: List[str],
    collection_name: str,
    extra_metadata: Optional[List[Dict[str, Any]]],
    enable_evolution: bool,
    llm_max_workers: int,
    chroma_batch_size: int,
    chroma_max_workers: int,
) -> Tuple[int, List[str]]:
    """写入一个批次，返回成功数和 note_ids。"""
    note_ids: List[str] = memory_system.add_notes_batch(
        contents=contents,
        category=collection_name,
        timestamps=None,
        extra_metadata=extra_metadata,
        enable_evolution=enable_evolution,
        llm_max_workers=llm_max_workers,
        chroma_batch_size=chroma_batch_size,
        chroma_max_workers=chroma_max_workers,
    )
    return len(note_ids), note_ids


def _process_entries(
    entries: List[Dict[str, Any]],
    collection_name: str,
    memory_system,
    batch_size: int,
    enable_evolution_from_batch: int,
    llm_max_workers: int,
    chroma_batch_size: int,
    chroma_max_workers: int,
    epoch_logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """按批写入，指定从第几个批次开始演化（1-based）。"""
    results = {
        "total": len(entries),
        "stored_successfully": 0,
        "failed_to_store": 0,
        "batch_details": [],
        "note_ids": [],
    }
    batches = _chunk(entries, batch_size)
    for idx, batch in enumerate(batches, start=1):
        # 记录 epoch 开始时间
        epoch_start_time = time.time()
        enable_evo = idx >= enable_evolution_from_batch
        contents: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for item in batch:
            contents.append(item["content"])
            metadata.append(
                {
                    "bullet_id": item.get("id"),
                    "retrieval_count": item.get("helpful_votes"),
                    "section": item.get("section"),
                    "source": item.get("source"),
                }
            )
        logger.info(
            "Batch %d/%d size=%d enable_evolution=%s",
            idx,
            len(batches),
            len(batch),
            enable_evo,
        )
        try:
            stored, note_ids = _store_batch(
                memory_system,
                contents,
                collection_name,
                metadata,
                enable_evo,
                llm_max_workers,
                chroma_batch_size,
                chroma_max_workers,
            )
            # 记录 epoch 结束时间并计算耗时
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # 控制台输出
            logger.info(
                "Epoch %d/%d completed in %.2f seconds (%.2f minutes)",
                idx,
                len(batches),
                epoch_duration,
                epoch_duration / 60.0,
            )
            
            # 文件只记录 epoch 时间
            if epoch_logger:
                epoch_logger.info(
                    "Epoch %d/%d completed in %.2f seconds (%.2f minutes)",
                    idx,
                    len(batches),
                    epoch_duration,
                    epoch_duration / 60.0,
                )
                # 立即刷新文件日志，确保实时写入
                for handler in epoch_logger.handlers:
                    handler.flush()
            results["stored_successfully"] += stored
            results["note_ids"].extend(note_ids)
            results["batch_details"].append(
                {
                    "batch_index": idx,
                    "batch_size": len(batch),
                    "enable_evolution": enable_evo,
                    "stored": stored,
                    "duration_seconds": round(epoch_duration, 2),
                }
            )
        except Exception as e:
            # 记录失败时的 epoch 时间
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # 控制台输出
            logger.error(
                "Batch %d failed after %.2f seconds: %s",
                idx,
                epoch_duration,
                e,
            )
            
            # 文件只记录 epoch 时间
            if epoch_logger:
                epoch_logger.error(
                    "Epoch %d/%d failed after %.2f seconds: %s",
                    idx,
                    len(batches),
                    epoch_duration,
                    str(e),
                )
                # 立即刷新文件日志，确保实时写入
                for handler in epoch_logger.handlers:
                    handler.flush()
            results["failed_to_store"] += len(batch)
            results["batch_details"].append(
                {
                    "batch_index": idx,
                    "batch_size": len(batch),
                    "enable_evolution": enable_evo,
                    "error": str(e),
                    "duration_seconds": round(epoch_duration, 2),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import ACE experiences and cases into A-mem-sys with evolution enabled after the first batch."
    )
    parser.add_argument("--report_path", required=True, help="Path to ACE questions_report_*.md")
    parser.add_argument("--mem_sys_path", required=True, help="Path to A-mem-sys root (where agentic_memory lives)")
    parser.add_argument("--memory_root", required=True, help="Root dir for A-mem-sys storage (e.g. memory_db)")
    parser.add_argument("--api_key", default=None, help="API key (fallback YUNWU_API_KEY env var)")
    parser.add_argument("--experience_collection_name", default="ace_experiences", help="Collection for experiences")
    parser.add_argument("--case_collection_name", default="ace_case_library", help="Collection for cases")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size; first batch no evolution")
    parser.add_argument("--evo_threshold", type=int, default=1000, help="evo_threshold passed to AgenticMemorySystem")
    parser.add_argument("--llm_backend", default="openai")
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--llm_max_workers", type=int, default=128, help="Parallelism for LLM analysis")
    parser.add_argument("--chroma_batch_size", type=int, default=256)
    parser.add_argument("--chroma_max_workers", type=int, default=64)
    parser.add_argument(
        "--evolution_from_batch",
        type=int,
        default=2,
        help="Enable evolution starting from this batch index (1-based). Default: 2 (first batch off).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file. If not specified, defaults to /gpfs/flash/home/wyj/futong/reports/ace_amem_integration_YYYYMMDD_HHMMSS.log",
    )

    args = parser.parse_args()

    # 配置日志：控制台输出所有日志，文件只记录 epoch 时间
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # 控制台处理器：输出所有日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[console_handler],
        force=True,
    )
    
    # 文件处理器：只记录 epoch 时间（如果未指定，使用默认路径）
    if args.log_file is None:
        # 默认日志目录
        default_log_dir = "/gpfs/flash/home/wyj/futong/reports"
        os.makedirs(default_log_dir, exist_ok=True)
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(default_log_dir, f"ace_amem_integration_{timestamp}.log")
    
    # 确保日志目录存在
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建专门的 epoch logger，只输出到文件
    epoch_logger = logging.getLogger("epoch_logger")
    epoch_logger.setLevel(logging.INFO)
    epoch_logger.propagate = False  # 不传播到根 logger，避免重复输出
    
    file_handler = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    epoch_logger.addHandler(file_handler)
    
    logger.info("Epoch time logging to file: %s", args.log_file)

    api_key = args.api_key or os.getenv("YUNWU_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Set YUNWU_API_KEY env var or pass --api_key.")

    # 1. 提取Experience/Case
    exp_extractor = ACEExperienceExtractor(args.report_path)
    experiences = exp_extractor.extract_experiences()
    case_extractor = CaseLibraryExtractor(args.report_path)
    cases = case_extractor.extract_cases()

    if not experiences and not cases:
        logger.error("Nothing to import (no experiences and no cases).")
        return

    summary: Dict[str, Any] = {}

    # 2. Experience导入：首批不演化，后续演化
    if experiences:
        logger.info("=== Import experiences (total=%d) ===", len(experiences))
        exp_ms = _build_memory_system(
            mem_sys_path=args.mem_sys_path,
            memory_root=args.memory_root,
            collection_name=args.experience_collection_name,
            api_key=api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            evo_threshold=args.evo_threshold,
        )
        exp_results = _process_entries(
            entries=experiences,
            collection_name=args.experience_collection_name,
            memory_system=exp_ms,
            batch_size=args.batch_size,
            enable_evolution_from_batch=args.evolution_from_batch,
            llm_max_workers=args.llm_max_workers,
            chroma_batch_size=args.chroma_batch_size,
            chroma_max_workers=args.chroma_max_workers,
            epoch_logger=epoch_logger,
        )
        summary["experience_import"] = exp_results
        logger.info("Experience import summary: %s", json.dumps(exp_results, ensure_ascii=False))

    # 3. Case导入：同策略
    if cases:
        logger.info("=== Import cases (total=%d) ===", len(cases))
        case_ms = _build_memory_system(
            mem_sys_path=args.mem_sys_path,
            memory_root=args.memory_root,
            collection_name=args.case_collection_name,
            api_key=api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            evo_threshold=args.evo_threshold,
        )
        case_results = _process_entries(
            entries=cases,
            collection_name=args.case_collection_name,
            memory_system=case_ms,
            batch_size=args.batch_size,
            enable_evolution_from_batch=args.evolution_from_batch,
            llm_max_workers=args.llm_max_workers,
            chroma_batch_size=args.chroma_batch_size,
            chroma_max_workers=args.chroma_max_workers,
            epoch_logger=epoch_logger,
        )
        summary["case_import"] = case_results
        logger.info("Case import summary: %s", json.dumps(case_results, ensure_ascii=False))

    logger.info("=== All done ===")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

