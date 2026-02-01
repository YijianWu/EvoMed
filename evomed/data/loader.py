# -*- coding: utf-8 -*-
"""
Batch Diagnosis Script

Batch runs multi-disciplinary diagnosis pipeline for cases in 1228.xlsx format.
"""

import json
import pandas as pd
import re
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from evomed.diagnosis import (
    DiagnosticPipeline, 
    PatientInfo,
    API_KEY, 
    API_BASE_URL
)


def parse_history_text(history_text: str, patient_id: str) -> PatientInfo:
    """
    Parses patient info from a single history text
    
    1228.xlsx format: Each row is a complete medical record text, containing:
    - Gender, Age
    - Chief Complaint/Symptoms
    - Past History
    - Physical Examination
    - Laboratory/Imaging Tests
    """
    if not isinstance(history_text, str) or not history_text.strip():
        return None
    
    text = history_text.strip()
    
    # Extract gender and age
    gender = "Unknown"
    age = 0
    
    # Match gender/age info
    gender_age_match = re.search(r'([FM])[\s,]\s*(\d+)', text, re.IGNORECASE)
    if gender_age_match:
        gender = "Female" if gender_age_match.group(1).upper() == "F" else "Male"
        age = int(gender_age_match.group(2))
    
    # Extract Chief Complaint (usually at the beginning, describing main symptoms)
    chief_complaint = ""
    # Try extracting the first sentence as Chief Complaint
    first_sentence = re.split(r'[.;]', text)[0] if text else ""
    if len(first_sentence) < 200:
        chief_complaint = first_sentence
    else:
        chief_complaint = first_sentence[:200] + "..."
    
    # Extract history of present illness (HPI)
    hpi = ""
    hpi_match = re.search(r'(?:HPI|History of Present Illness)[:]*(.+?)(?:Physical Examination|Past History|$)', text, re.DOTALL | re.IGNORECASE)
    if hpi_match:
        hpi = hpi_match.group(1).strip()
    
    # Extract Past History
    past_history = ""
    past_match = re.search(r'(?:Past History|Past Medical History)[:]*(.+?)(?:Physical Examination|$)', text, re.DOTALL | re.IGNORECASE)
    if past_match:
        past_history = past_match.group(1).strip()
    
    # Extract Physical Examination (PE)
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


def run_batch_diagnosis(
    excel_path: str = "1228.xlsx",
    output_dir: str = "batch_results",
    activation_mode: str = "eep_semantic",
    top_k: int = 5,
    enable_rag: bool = True,
    enable_experience: bool = True,
    enable_case: bool = True,
    start_index: int = 0,
    max_cases: int = None,
):
    """
    Runs batch diagnosis
    
    Args:
        excel_path: Input Excel file path
        output_dir: Output directory
        activation_mode: Expert activation mode
        top_k: Number of experts
        enable_rag: Enable RAG
        enable_experience: Enable experience library
        enable_case: Enable case library
        start_index: Start case index
        max_cases: Maximum number of cases to process (None for all)
    """
    print("=" * 70)
    print("Batch Multi-Disciplinary Diagnosis System")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"\nLoading case data: {excel_path}")
    df = pd.read_excel(excel_path)
    total_cases = len(df)
    print(f"Total {total_cases} cases")
    
    # Initialize diagnosis pipeline
    print("\nInitializing diagnosis pipeline...")
    pipeline = DiagnosticPipeline(
        activation_mode=activation_mode,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    # Determine range
    end_index = total_cases
    if max_cases:
        end_index = min(start_index + max_cases, total_cases)
    
    print(f"\nProcessing range: Case {start_index + 1} to {end_index}")
    print(f"Activation mode: {activation_mode} | Expert count: {top_k}")
    print("=" * 70)
    
    # Batch processing results summary
    summary = {
        "total": end_index - start_index,
        "success": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
        "cases": []
    }
    
    # Process cases one by one
    for idx in range(start_index, end_index):
        case_start = time.time()
        case_num = idx + 1
        
        print(f"\n{'='*70}")
        print(f"【Case {case_num}/{end_index}】")
        print("=" * 70)
        
        try:
            # Get history text
            history_text = df.iloc[idx].get('history', '')
            if pd.isna(history_text) or not str(history_text).strip():
                print(f"⚠️ Case {case_num} content is empty, skipping")
                summary["failed"] += 1
                continue
            
            # Parse patient information
            patient_id = f"case_{case_num:04d}"
            patient = parse_history_text(str(history_text), patient_id)
            
            if not patient:
                print(f"⚠️ Case {case_num} parsing failed, skipping")
                summary["failed"] += 1
                continue
            
            print(f"Patient: {patient.gender}, {patient.age} years old")
            print(f"Chief Complaint: {patient.chief_complaint[:50]}..." if len(patient.chief_complaint) > 50 else f"Chief Complaint: {patient.chief_complaint}")
            
            # Run diagnosis pipeline
            results = pipeline.run_pipeline(patient, top_k=top_k)
            
            # Save individual result
            output_file = os.path.join(output_dir, f"{patient_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            case_duration = time.time() - case_start
            print(f"\n✅ Case {case_num} completed, duration {case_duration:.1f}s, result saved to: {output_file}")
            
            summary["success"] += 1
            summary["cases"].append({
                "case_id": patient_id,
                "gender": patient.gender,
                "age": patient.age,
                "duration": round(case_duration, 1),
                "status": "success"
            })
            
        except Exception as e:
            print(f"\n❌ Case {case_num} processing failed: {e}")
            summary["failed"] += 1
            summary["cases"].append({
                "case_id": f"case_{case_num:04d}",
                "status": "failed",
                "error": str(e)
            })
    
    # Save summary report
    summary["end_time"] = datetime.now().isoformat()
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("Batch diagnosis complete!")
    print("=" * 70)
    print(f"Success: {summary['success']} / Failed: {summary['failed']} / Total: {summary['total']}")
    print(f"Summary report: {summary_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Multi-Disciplinary Diagnosis")
    parser.add_argument("--input", type=str, default="1228.xlsx", help="Input Excel file")
    parser.add_argument("--output", type=str, default="batch_results", help="Output directory")
    parser.add_argument("--mode", type=str, default="eep_semantic", help="Activation mode")
    parser.add_argument("--top_k", type=int, default=5, help="Expert count")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of cases")
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no_experience", action="store_true", help="Disable experience library")
    parser.add_argument("--no_case", action="store_true", help="Disable case library")
    
    args = parser.parse_args()
    
    run_batch_diagnosis(
        excel_path=args.input,
        output_dir=args.output,
        activation_mode=args.mode,
        top_k=args.top_k,
        start_index=args.start,
        max_cases=args.max,
        enable_rag=not args.no_rag,
        enable_experience=not args.no_experience,
        enable_case=not args.no_case,
    )
