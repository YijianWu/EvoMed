"""
遗传算法训练示例

演示如何使用遗传算法优化ExpertDiagnosis提示词
"""

import sys
sys.path.append('..')

from src.training.run_specialty_evolution import (
    EXPERT_TEMPLATES,
    run_specialty_evolution,
    load_patients_by_specialty
)
from src.main_diagnosis_pipeline import DiagnosticPipeline
import pandas as pd


def example_train_single_specialty():
    """单个Department的训练示例"""
    
    print("="*70)
    print("示例：单个Department遗传算法训练")
    print("="*70)
    
    # 1. 选择要训练的Department
    specialty = "肿瘤科"
    print(f"\n训练Department: {specialty}")
    
    # 2. 加载验证数据
    print("\n加载验证数据...")
    data_file = "../data/guilin_inpatient_extracted_10000.xlsx"
    try:
        df = pd.read_excel(data_file)
        patients = load_patients_by_specialty(df, specialty, count=5)
        print(f"  加载了 {len(patients)} 个验证Case")
    except FileNotFoundError:
        print(f"  ⚠️ 数据文件不存在: {data_file}")
        print("  请先准备数据文件（参见 data/README.md）")
        return
    
    # 3. 初始化Diagnosis流水线
    print("\n初始化Diagnosis流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",
        enable_rag=False,
        enable_experience=False,
        enable_case=False
    )
    
    # 4. 运行遗传算法优化
    print("\n开始遗传算法优化...")
    print("  - 种群大小: 16")
    print("  - 进化代数: 10")
    print("  - 返回Expert数: 2")
    
    experts = run_specialty_evolution(
        specialty=specialty,
        patients=patients,
        pipeline=pipeline,
        generations=10,
        population_size=16,
        top_k_return=2
    )
    
    # 5. 输出结果
    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    print(f"\n获得 {len(experts)} 个优化Expert:\n")
    
    for i, expert in enumerate(experts):
        print(f"[{i+1}] {expert['name']}")
        print(f"    ID: {expert['id']}")
        print(f"    Fitness: {expert['fitness']:.4f}")
        print(f"    准确率: {expert['stats']['accuracy']*100:.1f}%")
        print(f"    平均分: {expert['stats']['avg_score']:.1f}")
        print()


def example_customize_training_params():
    """自定义训练参数示例"""
    
    print("\n\n" + "="*70)
    print("示例：自定义训练参数")
    print("="*70)
    
    # 自定义遗传算法参数
    custom_config = {
        "generations": 15,        # 增加进化代数
        "population_size": 24,    # 增大种群规模
        "elitism_count": 3,       # 保留更多精英
        "mutation_rate": 0.3,     # 调整变异率
        "crossover_rate": 0.7     # 调整交叉率
    }
    
    print("\n自定义配置:")
    for key, value in custom_config.items():
        print(f"  - {key}: {value}")
    
    print("\n说明:")
    print("  1. 增加进化代数可能获得更好的结果，但耗时更长")
    print("  2. 增大种群规模可增加多样性，但计算成本更高")
    print("  3. 精英保留数影响收敛速度")
    print("  4. 变异率和交叉率需要平衡探索与利用")


def example_training_workflow():
    """完整训练流程示例"""
    
    print("\n\n" + "="*70)
    print("示例：完整14Department训练流程")
    print("="*70)
    
    print("""
    完整训练流程:
    
    1. 准备数据
       - 确保数据文件存在: data/guilin_inpatient_extracted_10000.xlsx
       - 数据格式符合要求（见 data/README.md）
    
    2. 运行完整训练
       ```bash
       python src/training/run_specialty_evolution.py
       ```
    
    3. 中断后继续训练
       ```bash
       python src/training/continue_specialty_evolution.py
       ```
    
    4. 查看结果
       - 中间结果: specialty_pool_<Department名>.json
       - 最终结果: outputs/optimized_expert_pool_28.json
    
    5. 训练参数调优建议
       - 初次训练：generations=10, population_size=16
       - 精细优化：generations=15-20, population_size=24-32
       - 快速测试：generations=5, population_size=8
    
    6. 预计训练时间
       - 单个Department：约15-30分钟（取决于API速度）
       - 14个Department：约4-8小时
       - 使用断点续训可随时中断和恢复
    """)


if __name__ == "__main__":
    # 运行单Department训练示例
    # 注意：需要先准备数据文件
    # example_train_single_specialty()
    
    # 查看自定义参数示例
    example_customize_training_params()
    
    # 查看完整训练流程
    example_training_workflow()


