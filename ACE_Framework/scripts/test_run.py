#!/usr/bin/env python3
"""测试运行 ACE 流程的简化脚本。

使用 DummyLLM 来快速验证整个数据流程是否正常。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
    Curator,
    Generator,
    OfflineAdapter,
    Playbook,
    Reflector,
    DummyLLMClient,
)
from ace.retrieval import SemanticRetriever


def main():
    parser = argparse.ArgumentParser(description="ACE Test Run with Dummy LLM")
    parser.add_argument("--excel", required=True, help="Path to input Excel file")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--output", default="test_output.json", help="Output file")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading data from {args.excel}...")
    print(f"[INFO] Reading first {args.limit} rows (use nrows for faster loading)...")
    
    # Use nrows to only read what we need - much faster for large files!
    df = pd.read_excel(args.excel, nrows=args.limit)
    print(f"[INFO] Successfully loaded {len(df)} rows")
    
    print(f"[INFO] Total samples: {len(df)}")
    
    # 初始化组件（使用 Dummy LLM）
    print("[INFO] Initializing components with DummyLLM...")
    dummy_client = DummyLLMClient()
    
    # 为每个样本预先队列化足够的响应
    # Reflector 需要一个 JSON 响应
    # Curator 也需要一个 JSON 响应
    for i in range(args.limit * 10):  # 每个样本可能需要多次调用（重试等）
        dummy_client.queue(json.dumps({
            "reasoning": "Test reasoning",
            "error_identification": "Test error",
            "root_cause_analysis": "Test analysis",
            "correct_approach": "Test approach",
            "key_insight": "Test insight",
            "bullet_tags": []
        }, ensure_ascii=False))
        
        dummy_client.queue(json.dumps({
            "operations": [{
                "type": "ADD",
                "section": "Test Section",
                "content": f"Test experience {i}",
                "bullet_id": None
            }]
        }, ensure_ascii=False))
    
    generator = Generator(llm=None)
    reflector = Reflector(dummy_client)
    curator = Curator(dummy_client)
    
    # 初始化 Playbook 和 Retriever
    print("[INFO] Initializing playbook and retriever...")
    playbook = Playbook()
    retriever = SemanticRetriever(lazy_load=True)
    
    # 处理样本
    print(f"[INFO] Processing {len(df)} samples...")
    results = []
    
    for idx, row in enumerate(df.itertuples(), 1):
        print(f"\n[INFO] Processing sample {idx}/{len(df)}")
        
        # 构建问题
        question_parts = []
        for field in ["性别_clean", "年龄_clean", "病历_clean"]:
            try:
                val = getattr(row, field, None)
                if pd.notna(val):
                    question_parts.append(f"{field}: {val}")
            except:
                pass
        
        question = "\n".join(question_parts)
        ground_truth = getattr(row, "诊断", "Unknown")
        
        print(f"  Question length: {len(question)} chars")
        print(f"  Ground truth: {ground_truth}")
        
        # 生成
        generator_output = generator.generate(
            question=question,
            context="Medical case",
            playbook=playbook,
            most_likely_diagnosis=ground_truth,
            diagnostic_rationale=f"Based on: {ground_truth}",
            bullet_ids=[]
        )
        
        print(f"  Generator output: {generator_output.final_answer[:50]}...")
        
        # 反思
        reflection = reflector.reflect(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback="Test feedback",
            max_refinement_rounds=1,
            playbook_excerpt="",
            allowed_ids=[]
        )
        
        print(f"  Reflection complete")
        
        # 策展
        curator_output = curator.curate(
            reflection=reflection,
            playbook=playbook,
            question_context=f"Sample {idx}",
            progress=f"{idx}/{len(df)}",
            playbook_text=""
        )
        
        print(f"  Curator generated {len(curator_output.delta.operations)} operations")
        
        # 应用 delta
        playbook.apply_delta(curator_output.delta)
        
        # 更新检索器
        for op in curator_output.delta.operations:
            if str(op.type).upper() == "ADD" and op.content:
                if op.bullet_id:
                    retriever.add_experience(op.bullet_id, op.content)
        
        results.append({
            "sample_id": f"q{idx:05d}",
            "question_length": len(question),
            "ground_truth": ground_truth,
            "operations": len(curator_output.delta.operations)
        })
    
    # 保存结果
    print(f"\n[INFO] Final playbook has {len(playbook._bullets)} bullets")
    print(f"[INFO] Retriever has {len(retriever.content_cache)} cached items")
    
    # 导出
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "samples_processed": len(results),
        "playbook_bullets": len(playbook._bullets),
        "playbook_sections": len(playbook._sections),
        "results": results,
        "playbook": playbook.to_dict()
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Test complete! Output saved to {args.output}")
    print(f"[INFO] Processed {len(results)} samples successfully")
    print(f"[INFO] Playbook stats: {playbook.stats()}")


if __name__ == "__main__":
    main()

