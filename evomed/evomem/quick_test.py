#!/usr/bin/env python3
"""简单的ACE流程测试"""
import sys, os, json
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '.')

print("=== START ===", flush=True)

try:
    import pandas as pd
    from ace import Playbook, Generator, Reflector, Curator, DummyLLMClient
    
    print("Imports OK", flush=True)
    
    # Load data
    df = pd.read_excel("guilin_100K_20K_诊断_20260102_161948.xlsx", nrows=1)
    print(f"Loaded {len(df)} rows", flush=True)
    
    # Setup
    dummy = DummyLLMClient()
    for i in range(10):
        dummy.queue(json.dumps({
            "reasoning": "test", "error_identification": "test",
            "root_cause_analysis": "test", "correct_approach": "test",
            "key_insight": f"insight{i}", "bullet_tags": []
        }))
        dummy.queue(json.dumps({"operations": [{
            "type": "ADD", "section": "Test", "content": f"exp{i}", "bullet_id": None
        }]}))
    
    gen = Generator(llm=None)
    ref = Reflector(dummy)
    cur = Curator(dummy)
    pb = Playbook()
    
    print("Components ready", flush=True)
    
    # Process
    row = df.iloc[0]
    diagnosis = row['诊断'] if '诊断' in df.columns else 'Unknown'
    print(f"Processing: {diagnosis}", flush=True)
    
    gen_out = gen.generate(
        question=f"Diagnosis: {diagnosis}",
        context="test",
        playbook=pb,
        most_likely_diagnosis=diagnosis,
        diagnostic_rationale="test"
    )
    print(f"Generator OK: {gen_out.final_answer}", flush=True)
    
    reflection = ref.reflect(
        question=f"Diagnosis: {diagnosis}",
        generator_output=gen_out,
        playbook=pb,
        ground_truth=diagnosis,
        feedback="test",
        max_refinement_rounds=1
    )
    print(f"Reflector OK: {reflection.key_insight[:30]}", flush=True)
    
    curator_out = cur.curate(
        reflection=reflection,
        playbook=pb,
        question_context="test",
        progress="1/1"
    )
    print(f"Curator OK: {len(curator_out.delta.operations)} ops", flush=True)
    
    pb.apply_delta(curator_out.delta)
    print(f"Applied to playbook: {len(pb.bullets())} bullets", flush=True)
    
    # Save
    with open('quick_test.json', 'w', encoding='utf-8') as f:
        json.dump({'status': 'success', 'bullets': len(pb.bullets())}, f, indent=2)
    
    print("=== SUCCESS ===", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()


