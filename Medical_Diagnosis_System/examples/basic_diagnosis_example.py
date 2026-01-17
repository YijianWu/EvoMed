"""
基础诊断示例

演示如何使用已优化的专家池进行医疗诊断
"""

import sys
sys.path.append('..')

from src.main_diagnosis_pipeline import DiagnosticPipeline, PatientInfo


def example_basic_diagnosis():
    """基础诊断示例"""
    
    print("="*70)
    print("示例：使用可演化专家池进行诊断")
    print("="*70)
    
    # 1. 初始化诊断流水线
    print("\n[1] 初始化诊断流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",  # 使用可演化专家池
        enable_rag=False,                # 可选：启用RAG检索
        enable_experience=False,
        enable_case=False
    )
    
    # 2. 构建患者信息
    print("\n[2] 构建患者信息...")
    patient = PatientInfo(
        patient_id="example_001",
        age=45,
        gender="女",
        chief_complaint="右上腹痛3天",
        present_illness="""
        患者3天前无明显诱因出现右上腹痛，呈持续性胀痛，
        向右肩背部放射，伴恶心、呕吐，进食油腻食物后症状加重。
        无发热、黄疸。
        """,
        past_history="既往体健，否认高血压、糖尿病史",
        physical_exam="""
        体温37.2℃，血压120/80mmHg，心率80次/分
        右上腹压痛阳性，墨菲氏征阳性
        肝脾未触及
        """,
        lab_results="WBC 12.5×10^9/L，中性粒细胞82%",
        diagnosis=""  # 待诊断
    )
    
    # 3. 执行诊断
    print("\n[3] 执行诊断流程...")
    print("  - Step 1: 科室路由")
    print("  - Step 2: 信息重构")
    print("  - Step 3: 专家诊断")
    print("  - Step 4: 结果聚合")
    
    result = pipeline.diagnose(patient)
    
    # 4. 输出结果
    print("\n" + "="*70)
    print("诊断结果")
    print("="*70)
    print(result)
    
    return result


def example_with_rag():
    """带RAG检索的诊断示例"""
    
    print("\n\n" + "="*70)
    print("示例：带RAG知识检索的诊断")
    print("="*70)
    
    # 启用RAG检索
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",
        enable_rag=True,  # 启用RAG
        enable_experience=False,
        enable_case=False
    )
    
    patient = PatientInfo(
        patient_id="example_002",
        age=28,
        gender="女",
        chief_complaint="停经40天，下腹痛伴阴道流血1天",
        present_illness="""
        患者平素月经规律，末次月经2个月前。
        1天前出现下腹痛，呈阵发性绞痛，伴少量阴道流血。
        无发热、恶心、呕吐。
        """,
        past_history="既往体健，G1P0",
        physical_exam="""
        体温36.8℃，血压110/70mmHg
        下腹压痛阳性，反跳痛阳性
        妇科检查：宫颈举痛阳性，附件区触痛
        """,
        lab_results="β-HCG 2000 mIU/mL",
        diagnosis=""
    )
    
    result = pipeline.diagnose(patient)
    
    print("\n诊断结果:")
    print(result)
    
    return result


if __name__ == "__main__":
    # 运行基础示例
    example_basic_diagnosis()
    
    # 运行RAG示例（如果RAG索引已构建）
    # example_with_rag()


