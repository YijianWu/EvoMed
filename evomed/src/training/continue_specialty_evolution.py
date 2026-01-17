"""
继续运行剩余科室的专家池训练

从中断处继续，只运行尚未完成的科室
"""

import json
import pandas as pd
import os
import time
from typing import List, Dict

# 导入主流程模块
from main_diagnosis_pipeline import (
    DiagnosticPipeline,
    PatientInfo,
    parse_patient_from_row
)

from run_specialty_evolution import (
    EXPERT_TEMPLATES,
    SPECIALTIES,
    DEPT_ALIASES,
    load_patients_by_specialty,
    run_specialty_evolution
)


def get_completed_specialties() -> List[str]:
    """检查已完成的科室（通过中间文件判断）"""
    completed = []
    for specialty in SPECIALTIES:
        intermediate_file = f"specialty_pool_{specialty}.json"
        if os.path.exists(intermediate_file):
            # 检查文件内容是否有效
            try:
                with open(intermediate_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if len(data) > 0:
                        completed.append(specialty)
                        print(f"  ✅ {specialty}: 已完成 ({len(data)} 位专家)")
            except:
                print(f"  ⚠️ {specialty}: 文件损坏")
    return completed


def main():
    """继续训练剩余科室"""
    
    print("="*70)
    print("🔄 继续运行分科赛道遗传算法进化系统")
    print("="*70)
    
    # 检查已完成的科室
    print("\n📋 检查已完成的科室...")
    completed = get_completed_specialties()
    
    # 确定待处理的科室
    remaining = [s for s in SPECIALTIES if s not in completed]
    
    print(f"\n📊 进度统计:")
    print(f"  - 已完成: {len(completed)} 个科室")
    print(f"  - 待处理: {len(remaining)} 个科室")
    
    if len(remaining) == 0:
        print("\n✅ 所有科室已完成训练！")
        return merge_all_results()
    
    print(f"\n📋 待处理科室列表:")
    for i, s in enumerate(remaining):
        print(f"  [{i+1}] {s}")
    
    # 加载数据
    print("\n📂 正在加载患者数据...")
    data_file = "guilin_inpatient_extracted_10000.xlsx"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    df = pd.read_excel(data_file)
    print(f"共加载 {len(df)} 条患者记录")
    
    # 初始化诊断流水线
    print("\n🔧 正在初始化诊断流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",
        enable_rag=False,
        enable_experience=False,
        enable_case=False,
    )
    
    # 继续训练剩余科室
    all_experts = []
    failed_specialties = []
    
    start_idx = len(completed)
    
    for idx, specialty in enumerate(remaining):
        print(f"\n\n{'#'*70}")
        print(f"# 进度: [{start_idx + idx + 1}/{len(SPECIALTIES)}] {specialty}")
        print(f"{'#'*70}")
        
        # 获取该科室的验证病例
        patients = load_patients_by_specialty(df, specialty, count=5)
        print(f"  筛选到 {len(patients)} 个病例")
        
        if len(patients) < 3:
            print(f"  ⚠️ 病例数不足3个，跳过该科室")
            failed_specialties.append(specialty)
            continue
        
        # 运行进化
        try:
            experts = run_specialty_evolution(
                specialty=specialty,
                patients=patients,
                pipeline=pipeline,
                generations=10,
                population_size=16,
                top_k_return=2
            )
            all_experts.extend(experts)
            
            # 保存中间结果
            intermediate_file = f"specialty_pool_{specialty}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(experts, f, ensure_ascii=False, indent=2)
            print(f"  💾 中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            failed_specialties.append(specialty)
            continue
        
        # 短暂休息，避免API限流
        time.sleep(2)
    
    # 合并所有结果
    print("\n\n" + "="*70)
    print("🏆 合并最终专家池")
    print("="*70)
    
    return merge_all_results()


def merge_all_results():
    """合并所有科室的专家池"""
    all_experts = []
    
    print("\n📂 读取所有科室的专家池...")
    for specialty in SPECIALTIES:
        intermediate_file = f"specialty_pool_{specialty}.json"
        if os.path.exists(intermediate_file):
            try:
                with open(intermediate_file, 'r', encoding='utf-8') as f:
                    experts = json.load(f)
                    all_experts.extend(experts)
                    print(f"  ✅ {specialty}: {len(experts)} 位专家")
            except Exception as e:
                print(f"  ❌ {specialty}: 读取失败 - {e}")
    
    # 按fitness排序
    all_experts.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    
    # 保存最终结果
    output_file = "optimized_expert_pool_28.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 最终专家池已保存: {output_file}")
    print(f"   共 {len(all_experts)} 位专家")
    
    # 汇总报告
    print("\n🎉 专家池分布:")
    specialty_counts = {}
    for expert in all_experts:
        s = expert.get('specialty', '未知')
        specialty_counts[s] = specialty_counts.get(s, 0) + 1
    
    for s, c in specialty_counts.items():
        print(f"  - {s}: {c} 位专家")
    
    return all_experts


if __name__ == "__main__":
    main()

