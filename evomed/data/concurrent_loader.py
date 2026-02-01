# -*- coding: utf-8 -*-
"""
Concurrent Batch Diagnosis Script

Supports multi-threaded concurrent diagnosis and outputs consolidated results to Excel.
"""

import json
import pandas as pd
import re
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from evomed.diagnosis import (
    DiagnosticPipeline, 
    PatientInfo,
    API_KEY, 
    API_BASE_URL
)

# Thread lock for printing
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe printing"""
    with print_lock:
        print(*args, **kwargs)


def parse_history_text(history_text: str, patient_id: str) -> Optional[PatientInfo]:
    """Parses patient info from a single history text"""
    if not isinstance(history_text, str) or not history_text.strip():
        return None
    
    text = history_text.strip()
    
    # Extract gender and age
    gender_age_match = re.search(r'([FM])[\s,]\s*(\d+)', text, re.IGNORECASE)
    if gender_age_match:
        gender = "Female" if gender_age_match.group(1).upper() == "F" else "Male"
        age = int(gender_age_match.group(2))
    
    # Extract chief complaint
    chief_complaint = ""
    first_sentence = re.split(r'[.;]', text)[0] if text else ""
    chief_complaint = first_sentence[:200] if len(first_sentence) > 200 else first_sentence
    
    # Extract history of present illness (HPI)
    hpi = ""
    hpi_match = re.search(r'(?:HPI|History of Present Illness)[:]*(.+?)(?:Physical Examination|Past History|$)', text, re.DOTALL | re.IGNORECASE)
    if hpi_match:
        hpi = hpi_match.group(1).strip()
    
    # Extract past history
    past_history = ""
    past_match = re.search(r'(?:Past History|Past Medical History)[:]*(.+?)(?:Physical Examination|$)', text, re.DOTALL | re.IGNORECASE)
    if past_match:
        past_history = past_match.group(1).strip()
    
    # Extract physical examination (PE)
    physical_exam = ""
    pe_match = re.search(r'(?:Physical Examination|PE)[:]*(.+?)(?:心脏冠状动脉|CT|Imaging|Labs|$)', text, re.DOTALL | re.IGNORECASE)
    if pe_match:
        physical_exam = pe_match.group(1).strip()
    
    # Extract laboratory tests
    labs = ""
    labs_match = re.search(r'(?:Labs|Laboratory Tests)[：:]*(.+)', text, re.DOTALL | re.IGNORECASE)
    if labs_match:
        labs = labs_match.group(1).strip()
    
    # Extract imaging tests
    imaging = ""
    img_match = re.search(r'(?:Imaging Tests|Imaging)[：:]*(.+?)(?:Labs|$)', text, re.DOTALL | re.IGNORECASE)
    if img_match:
        imaging = img_match.group(1).strip()
    
    return PatientInfo(
        patient_id=patient_id,
        gender=gender,
        age=age,
        department="Outpatient",
        chief_complaint=chief_complaint,
        history_of_present_illness=hpi,
        past_history=past_history,
        personal_history="",
        physical_examination=physical_exam,
        labs=labs,
        imaging=imaging,
        main_diagnosis="Pending Diagnosis",
        main_diagnosis_icd=""
    )


def extract_diagnosis_summary(results: Dict) -> Dict[str, Any]:
    """Extracts summary info from diagnosis results for Excel output"""
    summary = {
        "patient_id": results.get("patient_id", ""),
        "activation_mode": results.get("activation_mode", ""),
        "step1_route": "",
        "activated_experts": "",
        "final_diagnosis": "",
        "risk_warnings": "",
        "next_steps": "",
    }
    
    steps = results.get("steps", {})
    
    # Step-1 Routing Suggestion
    if "step1" in steps:
        step1_output = steps["step1"].get("output", "")
        # Extract first 500 characters as summary
        summary["step1_route"] = step1_output[:500] + "..." if len(step1_output) > 500 else step1_output
    
    # Activated Experts
    if "routing" in steps:
        experts = steps["routing"].get("selected_experts", [])
        summary["activated_experts"] = ", ".join(experts)
    
    # Step-4 Comprehensive Diagnosis
    if "step4" in steps:
        step4_output = steps["step4"].get("output", "")
        summary["final_diagnosis"] = step4_output[:1000] + "..." if len(step4_output) > 1000 else step4_output
        
        # Try to extract risk warnings
        risk_match = re.search(r'Risk[^：:]*[：:](.+?)(?:Next|Suggestion|$)', step4_output, re.DOTALL | re.IGNORECASE)
        if risk_match:
            summary["risk_warnings"] = risk_match.group(1).strip()[:300]
        
        # Try to extract next steps
        next_match = re.search(r'(?:Next Step|Suggestion)[^：:]*[：:](.+?)(?:$)', step4_output, re.DOTALL | re.IGNORECASE)
        if next_match:
            summary["next_steps"] = next_match.group(1).strip()[:300]
    
    return summary


def process_single_case(
    idx: int,
    history_text: str,
    pipeline: DiagnosticPipeline,
    top_k: int
) -> Dict[str, Any]:
    """Processes a single case (for concurrency)"""
    case_num = idx + 1
    patient_id = f"case_{case_num:04d}"
    
    result = {
        "case_num": case_num,
        "patient_id": patient_id,
        "status": "pending",
        "gender": "",
        "age": 0,
        "duration": 0,
        "error": "",
        "diagnosis_result": None,
        "summary": None
    }
    
    start_time = time.time()
    
    try:
        if pd.isna(history_text) or not str(history_text).strip():
            result["status"] = "skipped"
            result["error"] = "Empty content"
            return result
        
        # Parse patient information
        patient = parse_history_text(str(history_text), patient_id)
        if not patient:
            result["status"] = "failed"
            result["error"] = "Parsing failed"
            return result
        
        result["gender"] = patient.gender
        result["age"] = patient.age
        
        safe_print(f"[{case_num}] Starting diagnosis: {patient.gender}, {patient.age} years old")
        
        # Run diagnosis
        diagnosis_result = pipeline.run_pipeline(patient, top_k=top_k)
        
        result["status"] = "success"
        result["diagnosis_result"] = diagnosis_result
        result["summary"] = extract_diagnosis_summary(diagnosis_result)
        
        duration = time.time() - start_time
        result["duration"] = round(duration, 1)
        
        safe_print(f"[{case_num}] ✅ Completed, duration {duration:.1f}s")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["duration"] = round(time.time() - start_time, 1)
        safe_print(f"[{case_num}] ❌ Failed: {e}")
    
    return result


def run_concurrent_diagnosis(
    excel_path: str = "1228.xlsx",
    output_excel: str = "diagnosis_results.xlsx",
    output_json_dir: str = "batch_results",
    activation_mode: str = "eep_semantic",
    top_k: int = 5,
    max_workers: int = 3,
    max_cases: int = 20,
    enable_rag: bool = True,
    enable_experience: bool = True,
    enable_case: bool = True,
):
    """
    Runs concurrent batch diagnosis
    
    Args:
        excel_path: Input Excel file
        output_excel: Output Excel file
        output_json_dir: JSON results directory
        activation_mode: Activation mode
        top_k: Number of experts
        max_workers: Concurrent thread count
        max_cases: Maximum number of cases
    """
    print("=" * 70)
    print("Concurrent Batch Multi-Disciplinary Diagnosis System")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_json_dir, exist_ok=True)
    
    # Read data
    print(f"\nLoading case data: {excel_path}")
    df = pd.read_excel(excel_path)
    total_cases = min(len(df), max_cases)
    print(f"Will process {total_cases} cases (out of {len(df)})")
    
    # Initialize diagnosis pipeline
    print("\nInitializing diagnosis pipeline...")
    pipeline = DiagnosticPipeline(
        activation_mode=activation_mode,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    print(f"\nConcurrency: {max_workers} | Activation Mode: {activation_mode} | Expert Count: {top_k}")
    print("=" * 70)
    
    start_time = time.time()
    results = []
    
    # Concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx in range(total_cases):
            history_text = df.iloc[idx].get('history', '')
            future = executor.submit(
                process_single_case,
                idx, history_text, pipeline, top_k
            )
            futures[future] = idx
        
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Save individual JSON result
            if result["diagnosis_result"]:
                json_path = os.path.join(output_json_dir, f"{result['patient_id']}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result["diagnosis_result"], f, ensure_ascii=False, indent=2)
    
    # Sort by case number
    results.sort(key=lambda x: x["case_num"])
    
    total_duration = time.time() - start_time
    
    # Generate Excel summary
    print("\nGenerating Excel summary...")
    
    excel_data = []
    for r in results:
        row = {
            "Case No.": r["case_num"],
            "Patient ID": r["patient_id"],
            "Gender": r["gender"],
            "Age": r["age"],
            "Status": r["status"],
            "Duration (sec)": r["duration"],
            "Error Message": r.get("error", ""),
        }
        
        if r["summary"]:
            row["Activated Experts"] = r["summary"].get("activated_experts", "")
            row["Comprehensive Diagnosis"] = r["summary"].get("final_diagnosis", "")
            row["Risk Warnings"] = r["summary"].get("risk_warnings", "")
            row["Next Step Suggestions"] = r["summary"].get("next_steps", "")
        
        excel_data.append(row)
    
    result_df = pd.DataFrame(excel_data)
    result_df.to_excel(output_excel, index=False, engine='openpyxl')
    
    # Statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print("\n" + "=" * 70)
    print("Batch diagnosis complete!")
    print("=" * 70)
    print(f"Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Average per case: {total_duration/total_cases:.1f}s")
    print(f"\nExcel results: {output_excel}")
    print(f"JSON results directory: {output_json_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concurrent Batch Multi-Disciplinary Diagnosis")
    parser.add_argument("--input", type=str, default="1228.xlsx", help="Input Excel")
    parser.add_argument("--output", type=str, default="diagnosis_results.xlsx", help="Output Excel")
    parser.add_argument("--json_dir", type=str, default="batch_results", help="JSON directory")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent thread count")
    parser.add_argument("--max", type=int, default=20, help="Maximum number of cases")
    parser.add_argument("--top_k", type=int, default=5, help="Expert count")
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no_experience", action="store_true", help="Disable experience library")
    parser.add_argument("--no_case", action="store_true", help="Disable case library")
    
    args = parser.parse_args()
    
    run_concurrent_diagnosis(
        excel_path=args.input,
        output_excel=args.output,
        output_json_dir=args.json_dir,
        max_workers=args.workers,
        max_cases=args.max,
        top_k=args.top_k,
        enable_rag=not args.no_rag,
        enable_experience=not args.no_experience,
        enable_case=not args.no_case,
    )
