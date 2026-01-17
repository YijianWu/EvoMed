# -*- coding: utf-8 -*-
"""
并发批量诊断脚本

支持多线程并发运行诊断，输出Excel汇总结果
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

from main_diagnosis_pipeline import (
    DiagnosticPipeline, 
    PatientInfo,
    API_KEY, 
    API_BASE_URL
)

# 线程锁用于打印
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """线程安全的打印"""
    with print_lock:
        print(*args, **kwargs)


def parse_history_text(history_text: str, patient_id: str) -> Optional[PatientInfo]:
    """从单一history文本中解析患者信息"""
    if not isinstance(history_text, str) or not history_text.strip():
        return None
    
    text = history_text.strip()
    
    # 提取性别和年龄
    gender = "未知"
    age = 0
    
    gender_age_match = re.search(r'([男女])[，,]\s*(\d+)岁', text)
    if gender_age_match:
        gender = gender_age_match.group(1)
        age = int(gender_age_match.group(2))
    
    # 提取主诉
    chief_complaint = ""
    first_sentence = re.split(r'[。；]', text)[0] if text else ""
    chief_complaint = first_sentence[:200] if len(first_sentence) > 200 else first_sentence
    
    # 提取现病史
    hpi = ""
    hpi_match = re.search(r'岁[，,](.+?)(?:查体|体格检查|高血压病史|既往)', text, re.DOTALL)
    if hpi_match:
        hpi = hpi_match.group(1).strip()
    
    # 提取既往史
    past_history = ""
    past_match = re.search(r'(高血压病史|糖尿病病史|既往).+?(?:查体|体格检查|$)', text, re.DOTALL)
    if past_match:
        past_history = past_match.group(0).strip()
        if "查体" in past_history:
            past_history = past_history.split("查体")[0].strip()
    
    # 提取体格检查
    physical_exam = ""
    pe_match = re.search(r'(?:查体|体格检查)[：:]*(.+?)(?:心脏冠状动脉|CT|白细胞|超敏C反应蛋白|$)', text, re.DOTALL)
    if pe_match:
        physical_exam = pe_match.group(1).strip()
    
    # 提取实验室检查
    labs = ""
    labs_match = re.search(r'(白细胞|超敏C反应蛋白|血常规|生化).+', text, re.DOTALL)
    if labs_match:
        labs = labs_match.group(0).strip()
    
    # 提取影像学检查
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
        main_diagnosis="待诊断",
        main_diagnosis_icd=""
    )


def extract_diagnosis_summary(results: Dict) -> Dict[str, Any]:
    """从诊断结果中提取摘要信息用于Excel输出"""
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
    
    # Step-1 路由建议
    if "step1" in steps:
        step1_output = steps["step1"].get("output", "")
        # 提取前500字符作为摘要
        summary["step1_route"] = step1_output[:500] + "..." if len(step1_output) > 500 else step1_output
    
    # 激活的专家
    if "routing" in steps:
        experts = steps["routing"].get("selected_experts", [])
        summary["activated_experts"] = ", ".join(experts)
    
    # Step-4 综合诊断
    if "step4" in steps:
        step4_output = steps["step4"].get("output", "")
        summary["final_diagnosis"] = step4_output[:1000] + "..." if len(step4_output) > 1000 else step4_output
        
        # 尝试提取风险警示
        risk_match = re.search(r'风险[^：:]*[：:](.+?)(?:下一步|建议|$)', step4_output, re.DOTALL)
        if risk_match:
            summary["risk_warnings"] = risk_match.group(1).strip()[:300]
        
        # 尝试提取下一步建议
        next_match = re.search(r'(?:下一步|建议)[^：:]*[：:](.+?)(?:$)', step4_output, re.DOTALL)
        if next_match:
            summary["next_steps"] = next_match.group(1).strip()[:300]
    
    return summary


def process_single_case(
    idx: int,
    history_text: str,
    pipeline: DiagnosticPipeline,
    top_k: int
) -> Dict[str, Any]:
    """处理单个病例（用于并发）"""
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
            result["error"] = "内容为空"
            return result
        
        # 解析患者信息
        patient = parse_history_text(str(history_text), patient_id)
        if not patient:
            result["status"] = "failed"
            result["error"] = "解析失败"
            return result
        
        result["gender"] = patient.gender
        result["age"] = patient.age
        
        safe_print(f"[{case_num}] 开始诊断: {patient.gender}, {patient.age}岁")
        
        # 运行诊断
        diagnosis_result = pipeline.run_pipeline(patient, top_k=top_k)
        
        result["status"] = "success"
        result["diagnosis_result"] = diagnosis_result
        result["summary"] = extract_diagnosis_summary(diagnosis_result)
        
        duration = time.time() - start_time
        result["duration"] = round(duration, 1)
        
        safe_print(f"[{case_num}] ✅ 完成，耗时 {duration:.1f}s")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["duration"] = round(time.time() - start_time, 1)
        safe_print(f"[{case_num}] ❌ 失败: {e}")
    
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
    并发批量运行诊断
    
    Args:
        excel_path: 输入Excel文件
        output_excel: 输出Excel文件
        output_json_dir: JSON结果目录
        activation_mode: 激活模式
        top_k: 专家数量
        max_workers: 并发线程数
        max_cases: 最大病例数
    """
    print("=" * 70)
    print("并发批量多学科诊断系统")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(output_json_dir, exist_ok=True)
    
    # 读取数据
    print(f"\n正在加载病例数据: {excel_path}")
    df = pd.read_excel(excel_path)
    total_cases = min(len(df), max_cases)
    print(f"将处理 {total_cases} 个病例（共 {len(df)} 个）")
    
    # 初始化诊断流水线
    print("\n正在初始化诊断流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode=activation_mode,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    print(f"\n并发数: {max_workers} | 激活模式: {activation_mode} | 专家数量: {top_k}")
    print("=" * 70)
    
    start_time = time.time()
    results = []
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx in range(total_cases):
            history_text = df.iloc[idx].get('history', '')
            future = executor.submit(
                process_single_case,
                idx, history_text, pipeline, top_k
            )
            futures[future] = idx
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            # 保存单个JSON结果
            if result["diagnosis_result"]:
                json_path = os.path.join(output_json_dir, f"{result['patient_id']}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result["diagnosis_result"], f, ensure_ascii=False, indent=2)
    
    # 按病例号排序
    results.sort(key=lambda x: x["case_num"])
    
    total_duration = time.time() - start_time
    
    # 生成Excel汇总
    print("\n正在生成Excel汇总...")
    
    excel_data = []
    for r in results:
        row = {
            "病例号": r["case_num"],
            "患者ID": r["patient_id"],
            "性别": r["gender"],
            "年龄": r["age"],
            "状态": r["status"],
            "耗时(秒)": r["duration"],
            "错误信息": r.get("error", ""),
        }
        
        if r["summary"]:
            row["激活专家"] = r["summary"].get("activated_experts", "")
            row["综合诊断"] = r["summary"].get("final_diagnosis", "")
            row["风险警示"] = r["summary"].get("risk_warnings", "")
            row["下一步建议"] = r["summary"].get("next_steps", "")
        
        excel_data.append(row)
    
    result_df = pd.DataFrame(excel_data)
    result_df.to_excel(output_excel, index=False, engine='openpyxl')
    
    # 统计
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print("\n" + "=" * 70)
    print("批量诊断完成！")
    print("=" * 70)
    print(f"成功: {success_count} | 失败: {failed_count} | 跳过: {skipped_count}")
    print(f"总耗时: {total_duration:.1f}s ({total_duration/60:.1f}分钟)")
    print(f"平均每病例: {total_duration/total_cases:.1f}s")
    print(f"\nExcel结果: {output_excel}")
    print(f"JSON结果目录: {output_json_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="并发批量多学科诊断")
    parser.add_argument("--input", type=str, default="1228.xlsx", help="输入Excel")
    parser.add_argument("--output", type=str, default="diagnosis_results.xlsx", help="输出Excel")
    parser.add_argument("--json_dir", type=str, default="batch_results", help="JSON目录")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数")
    parser.add_argument("--max", type=int, default=20, help="最大病例数")
    parser.add_argument("--top_k", type=int, default=5, help="专家数量")
    parser.add_argument("--no_rag", action="store_true", help="禁用RAG")
    parser.add_argument("--no_experience", action="store_true", help="禁用经验库")
    parser.add_argument("--no_case", action="store_true", help="禁用病例库")
    
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

