# -*- coding: utf-8 -*-
"""
批量Diagnosis脚本

针对1228.xlsx格式的Case批量运行多学科Diagnosis流水线
"""

import json
import pandas as pd
import re
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from main_diagnosis_pipeline import (
    DiagnosticPipeline, 
    PatientInfo,
    API_KEY, 
    API_BASE_URL
)


def parse_history_text(history_text: str, patient_id: str) -> PatientInfo:
    """
    从单一history文本中解析Patient信息
    
    1228.xlsx格式: 每行是一段完整病历文本，包含：
    - Gender、Age
    - Chief Complaint/症状
    - Past History
    - Physical Examination
    - 实验室/影像检查
    """
    if not isinstance(history_text, str) or not history_text.strip():
        return None
    
    text = history_text.strip()
    
    # 提取Gender和Age
    gender = "未知"
    age = 0
    
    # 匹配 "女，50岁" 或 "男，41岁" 等格式
    gender_age_match = re.search(r'([男女])[，,]\s*(\d+)岁', text)
    if gender_age_match:
        gender = gender_age_match.group(1)
        age = int(gender_age_match.group(2))
    
    # 提取Chief Complaint（通常在开头，描述主要症状）
    chief_complaint = ""
    # 尝试提取第一句话作为Chief Complaint
    first_sentence = re.split(r'[。；]', text)[0] if text else ""
    if len(first_sentence) < 200:
        chief_complaint = first_sentence
    else:
        chief_complaint = first_sentence[:200] + "..."
    
    # 提取History of Present Illness（从Age后到"查体"或"检查"之前）
    hpi = ""
    hpi_match = re.search(r'岁[，,](.+?)(?:查体|Physical Examination|高血压病史|既往)', text, re.DOTALL)
    if hpi_match:
        hpi = hpi_match.group(1).strip()
    
    # 提取Past History
    past_history = ""
    past_match = re.search(r'(高血压病史|糖尿病病史|既往).+?(?:查体|Physical Examination|$)', text, re.DOTALL)
    if past_match:
        past_history = past_match.group(0).strip()
        # 截断到查体之前
        if "查体" in past_history:
            past_history = past_history.split("查体")[0].strip()
    
    # 提取Physical Examination
    physical_exam = ""
    pe_match = re.search(r'(?:查体|Physical Examination)[：:]*(.+?)(?:心脏冠状动脉|CT|白细胞|超敏C反应蛋白|$)', text, re.DOTALL)
    if pe_match:
        physical_exam = pe_match.group(1).strip()
    
    # 提取Laboratory Tests（从白细胞、CRP等开始）
    labs = ""
    labs_match = re.search(r'(白细胞|超敏C反应蛋白|血常规|生化).+', text, re.DOTALL)
    if labs_match:
        labs = labs_match.group(0).strip()
    
    # 提取Imaging Tests
    imaging = ""
    img_match = re.search(r'(CT|MRI|X线|超声|心电图).+?(?:白细胞|超敏C反应蛋白|$)', text, re.DOTALL)
    if img_match:
        imaging = img_match.group(0).strip()
    
    return PatientInfo(
        patient_id=patient_id,
        gender=gender,
        age=age,
        department="门诊",
        chief_complaint=chief_complaint,
        history_of_present_illness=hpi,
        past_history=past_history,
        personal_history="",
        physical_examination=physical_exam,
        labs=labs,
        imaging=imaging,
        main_diagnosis="待Diagnosis",
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
    批量运行Diagnosis
    
    Args:
        excel_path: 输入Excel文件路径
        output_dir: 输出目录
        activation_mode: Expert激活模式
        top_k: 激活Expert数量
        enable_rag: 启用RAG
        enable_experience: 启用Experience库
        enable_case: 启用Case库
        start_index: 起始Case索引
        max_cases: 最大处理Case数（None表示全部）
    """
    print("=" * 70)
    print("批量多学科Diagnosis系统")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"\n正在加载Case数据: {excel_path}")
    df = pd.read_excel(excel_path)
    total_cases = len(df)
    print(f"共 {total_cases} 个Case")
    
    # 初始化Diagnosis流水线
    print("\n正在初始化Diagnosis流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode=activation_mode,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    # 确定处理范围
    end_index = total_cases
    if max_cases:
        end_index = min(start_index + max_cases, total_cases)
    
    print(f"\n处理范围: Case {start_index + 1} 到 {end_index}")
    print(f"激活模式: {activation_mode} | Expert数量: {top_k}")
    print("=" * 70)
    
    # 批量处理结果汇总
    summary = {
        "total": end_index - start_index,
        "success": 0,
        "failed": 0,
        "start_time": datetime.now().isoformat(),
        "cases": []
    }
    
    # 逐个处理Case
    for idx in range(start_index, end_index):
        case_start = time.time()
        case_num = idx + 1
        
        print(f"\n{'='*70}")
        print(f"【Case {case_num}/{end_index}】")
        print("=" * 70)
        
        try:
            # 获取病历文本
            history_text = df.iloc[idx].get('history', '')
            if pd.isna(history_text) or not str(history_text).strip():
                print(f"⚠️ Case {case_num} 内容为空，跳过")
                summary["failed"] += 1
                continue
            
            # 解析Patient信息
            patient_id = f"case_{case_num:04d}"
            patient = parse_history_text(str(history_text), patient_id)
            
            if not patient:
                print(f"⚠️ Case {case_num} 解析失败，跳过")
                summary["failed"] += 1
                continue
            
            print(f"Patient: {patient.gender}, {patient.age}岁")
            print(f"Chief Complaint: {patient.chief_complaint[:50]}..." if len(patient.chief_complaint) > 50 else f"Chief Complaint: {patient.chief_complaint}")
            
            # 运行Diagnosis流水线
            results = pipeline.run_pipeline(patient, top_k=top_k)
            
            # 保存单个结果
            output_file = os.path.join(output_dir, f"{patient_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            case_duration = time.time() - case_start
            print(f"\n✅ Case {case_num} 完成，耗时 {case_duration:.1f}s，结果保存至: {output_file}")
            
            summary["success"] += 1
            summary["cases"].append({
                "case_id": patient_id,
                "gender": patient.gender,
                "age": patient.age,
                "duration": round(case_duration, 1),
                "status": "success"
            })
            
        except Exception as e:
            print(f"\n❌ Case {case_num} 处理失败: {e}")
            summary["failed"] += 1
            summary["cases"].append({
                "case_id": f"case_{case_num:04d}",
                "status": "failed",
                "error": str(e)
            })
    
    # 保存汇总报告
    summary["end_time"] = datetime.now().isoformat()
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("批量Diagnosis完成！")
    print("=" * 70)
    print(f"成功: {summary['success']} / 失败: {summary['failed']} / 总计: {summary['total']}")
    print(f"汇总报告: {summary_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量多学科Diagnosis")
    parser.add_argument("--input", type=str, default="1228.xlsx", help="输入Excel文件")
    parser.add_argument("--output", type=str, default="batch_results", help="输出目录")
    parser.add_argument("--mode", type=str, default="eep_semantic", help="激活模式")
    parser.add_argument("--top_k", type=int, default=5, help="Expert数量")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--max", type=int, default=None, help="最大Case数")
    parser.add_argument("--no_rag", action="store_true", help="禁用RAG")
    parser.add_argument("--no_experience", action="store_true", help="禁用Experience库")
    parser.add_argument("--no_case", action="store_true", help="禁用Case库")
    
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

