"""
分科赛道遗传算法进化脚本

针对14个科室分别训练专家池，每个科室：
1. 从数据集中筛选该科室的5个病例作为验证集
2. 基于该科室的专家模版进行10代GA进化
3. 取出Top 2个最优专家（包含完整的专家设定）

最终合并为28人专家池
"""

import json
import pandas as pd
import os
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 导入主流程模块
from main_diagnosis_pipeline import (
    DiagnosticPipeline,
    GeneticPromptOptimizer,
    PatientInfo,
    parse_patient_from_row
)
from system_step3_diag import system_step3_prompt


# 14个科室的完整专家定义模版
EXPERT_TEMPLATES = {
    "妇产科": {
        "id": "obgyn",
        "name": "妇产科专家",
        "specialty": "妇产科",
        "description": "擅长女性生殖系统疾病的诊治及围产期保健，关注女性全生命周期健康",
        "focus_areas": ["孕产期管理", "妇科肿瘤", "月经失调", "生殖道炎症"],
        "thinking_patterns": ["月经周期评估", "妊娠相关鉴别", "激素水平分析", "盆腔影像解读"],
    },
    "消化内科": {
        "id": "gastroenterology",
        "name": "消化内科专家",
        "specialty": "消化内科",
        "description": "擅长食管、胃肠、肝胆胰等消化系统疾病的内科诊治及内镜检查",
        "focus_areas": ["胃炎与溃疡", "肝脏疾病", "功能性胃肠病", "消化道出血"],
        "thinking_patterns": ["腹痛定位分析", "肝功能评估", "内镜指征判断", "幽门螺杆菌筛查"],
    },
    "儿科": {
        "id": "pediatrics",
        "name": "儿科专家",
        "specialty": "儿科",
        "description": "专注新生儿至青少年时期的生长发育及疾病诊治，关注儿童特有的生理病理特点",
        "focus_areas": ["呼吸道感染", "生长发育评估", "小儿消化系统", "新生儿护理"],
        "thinking_patterns": ["年龄相关鉴别", "生长曲线评估", "疫苗接种史", "喂养方式分析"],
    },
    "内分泌科": {
        "id": "endocrinology",
        "name": "内分泌科专家",
        "specialty": "内分泌科",
        "description": "擅长激素分泌异常及代谢性疾病的诊断与长期管理",
        "focus_areas": ["糖尿病管理", "甲状腺疾病", "骨质疏松", "肥胖与代谢综合征"],
        "thinking_patterns": ["血糖谱分析", "甲功五项解读", "胰岛功能评估", "代谢指标整合"],
    },
    "肝胆外科": {
        "id": "hepatobiliary_surgery",
        "name": "肝胆外科专家",
        "specialty": "肝胆外科",
        "description": "擅长肝脏、胆道及胰腺疾病的外科手术治疗及围手术期管理",
        "focus_areas": ["胆石症", "肝脏肿瘤", "胰腺炎", "胆道梗阻"],
        "thinking_patterns": ["黄疸鉴别", "肝功能储备评估", "手术指征判断", "影像学分期"],
    },
    "骨科": {
        "id": "orthopedics",
        "name": "骨科专家",
        "specialty": "骨科",
        "description": "擅长骨骼、关节、肌肉及韧带等运动系统疾病的诊断、复位及手术治疗",
        "focus_areas": ["骨折创伤", "关节炎", "脊柱疾病", "运动损伤"],
        "thinking_patterns": ["受伤机制分析", "X线骨折分型", "关节活动度评估", "神经损伤筛查"],
    },
    "呼吸内科": {
        "id": "respiratory",
        "name": "呼吸内科专家",
        "specialty": "呼吸内科",
        "description": "擅长呼吸系统感染、气道疾病及肺部肿瘤的内科诊断与治疗",
        "focus_areas": ["慢性阻塞性肺病", "哮喘管理", "肺部结节", "肺部感染"],
        "thinking_patterns": ["肺功能评估", "影像学征象解读", "感染vs肿瘤鉴别", "气道反应性分析"],
    },
    "急诊科": {
        "id": "emergency",
        "name": "急诊科专家",
        "specialty": "急诊科",
        "description": "擅长急性病、创伤及各类危重症的初步评估、急救复苏与分诊",
        "focus_areas": ["生命体征维持", "急性中毒", "多发伤", "心肺复苏"],
        "thinking_patterns": ["ABCDE评估", "危重症识别", "快速分诊", "时间敏感性判断"],
    },
    "泌尿外科": {
        "id": "urology",
        "name": "泌尿外科专家",
        "specialty": "泌尿外科",
        "description": "擅长泌尿系统（肾、膀胱、尿路）及男性生殖系统疾病的外科及微创治疗",
        "focus_areas": ["泌尿系结石", "前列腺疾病", "泌尿系肿瘤", "尿路感染"],
        "thinking_patterns": ["尿常规解读", "PSA分析", "结石成分分析", "排尿功能评估"],
    },
    "全科医学科": {
        "id": "general_practice",
        "name": "全科医学专家",
        "specialty": "全科医学科",
        "description": "提供综合性、连续性的基本医疗服务，擅长未分化疾病的初诊与常见病管理",
        "focus_areas": ["健康查体解读", "常见病初诊", "双向转诊", "慢病长期随访"],
        "thinking_patterns": ["整体评估", "风险分层", "预防医学", "多系统整合"],
    },
    "胃肠外科": {
        "id": "gastro_surgery",
        "name": "胃肠外科专家",
        "specialty": "胃肠外科",
        "description": "擅长胃、小肠、结直肠及肛门疾病的手术治疗，特别是肿瘤及急腹症处理",
        "focus_areas": ["胃肠道肿瘤", "阑尾炎", "肠梗阻", "疝气修补"],
        "thinking_patterns": ["急腹症鉴别", "肠梗阻分型", "肿瘤分期", "手术时机判断"],
    },
    "胸心血管外科": {
        "id": "cardiothoracic_surgery",
        "name": "胸心外科专家",
        "specialty": "胸心血管外科",
        "description": "擅长胸腔内器官（心脏、大血管、肺、食管）的复杂外科手术治疗",
        "focus_areas": ["肺癌手术", "心脏瓣膜病", "冠脉搭桥", "主动脉夹层"],
        "thinking_patterns": ["胸片CT解读", "心功能评估", "手术风险分层", "术前优化"],
    },
    "肿瘤科": {
        "id": "oncology",
        "name": "肿瘤科专家",
        "specialty": "肿瘤科",
        "description": "擅长各类良恶性肿瘤的内科综合治疗，包括化疗、靶向治疗及免疫治疗",
        "focus_areas": ["放化疗方案", "肿瘤筛查", "癌痛管理", "多学科会诊(MDT)"],
        "thinking_patterns": ["肿瘤标志物解读", "TNM分期", "基因突变分析", "预后评估"],
    },
    "心血管内科": {
        "id": "cardiology",
        "name": "心血管内科专家",
        "specialty": "心血管内科",
        "description": "擅长心脏及血管疾病的内科诊疗与介入治疗，关注心血管风险防控",
        "focus_areas": ["高血压", "冠心病", "心律失常", "心力衰竭"],
        "thinking_patterns": ["心电图分析", "心肌酶谱解读", "危险分层", "抗凝策略"],
    },
}

# 14个科室列表
SPECIALTIES = list(EXPERT_TEMPLATES.keys())

# 科室名称映射（处理数据中可能的别名）
DEPT_ALIASES = {
    "妇产科": ["妇产科", "妇科", "产科"],
    "消化内科": ["消化内科", "消化科"],
    "儿科": ["儿科", "小儿科", "儿内科"],
    "内分泌科": ["内分泌科", "内分泌"],
    "肝胆外科": ["肝胆外科", "肝胆胰外科"],
    "骨科": ["骨科", "骨外科", "创伤骨科"],
    "呼吸内科": ["呼吸内科", "呼吸科", "肺科"],
    "急诊科": ["急诊科", "急诊", "急诊医学科"],
    "泌尿外科": ["泌尿外科", "泌尿科"],
    "全科医学科": ["全科医学科", "全科", "全科医学", "综合内科"],
    "胃肠外科": ["胃肠外科", "普外科", "胃肠道外科"],
    "胸心血管外科": ["胸心血管外科", "胸外科", "心外科", "心胸外科"],
    "肿瘤科": ["肿瘤科", "肿瘤内科", "肿瘤外科"],
    "心血管内科": ["心血管内科", "心内科", "心脏科", "心脏内科"],
}


def load_patients_by_specialty(df: pd.DataFrame, specialty: str, count: int = 5) -> List[PatientInfo]:
    """
    从DataFrame中筛选指定科室的患者
    
    Args:
        df: 患者数据DataFrame
        specialty: 科室名称
        count: 需要的病例数量
    
    Returns:
        该科室的患者列表
    """
    aliases = DEPT_ALIASES.get(specialty, [specialty])
    
    # 筛选该科室的有效病例
    valid_mask = df['is_history_cleaned'] == True
    dept_mask = df['normalized_name'].apply(lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False)
    
    filtered_df = df[valid_mask & dept_mask]
    
    if len(filtered_df) == 0:
        print(f"  ⚠️ 科室 '{specialty}' 未找到有效病例，尝试模糊匹配...")
        # 尝试更宽松的匹配
        for alias in aliases:
            dept_mask = df['normalized_name'].str.contains(alias, na=False, case=False)
            filtered_df = df[valid_mask & dept_mask]
            if len(filtered_df) > 0:
                break
    
    patients = []
    for i in range(min(count, len(filtered_df))):
        try:
            p = parse_patient_from_row(filtered_df.iloc[i])
            patients.append(p)
        except Exception as e:
            print(f"    解析患者失败: {e}")
            continue
    
    return patients


class SpecialtyPromptOptimizer(GeneticPromptOptimizer):
    """
    专科定向Prompt优化器
    
    继承自通用优化器，但：
    1. 使用该科室的专家模版进行评估
    2. 产出包含完整专家设定的结构化数据
    """
    
    def __init__(self, specialty: str, expert_template: Dict, 
                 base_prompt: str, pipeline: DiagnosticPipeline,
                 population_size: int = 16, elitism_count: int = 2):
        super().__init__(base_prompt, pipeline, population_size, elitism_count)
        self.specialty = specialty
        self.expert_template = expert_template
        
    def _get_matched_expert(self, patient: PatientInfo) -> Dict:
        """重写：始终返回该科室的专家模版"""
        return self.expert_template
    
    def _mutate_prompt(self, prompt_text: str, intensity: str = "medium", 
                      max_retries: int = 2) -> str:
        """重写：变异时融入专科特色"""
        mutation_prompt = f"""
你是一个医疗AI Prompt优化专家。请对以下用于【{self.specialty}】诊断的 System Prompt 进行【变异】操作。

【该科室专家特征】
- 名称: {self.expert_template['name']}
- 专业: {self.expert_template['specialty']}
- 描述: {self.expert_template['description']}
- 关注领域: {', '.join(self.expert_template['focus_areas'])}
- 思维模式: {', '.join(self.expert_template.get('thinking_patterns', []))}

【原始 Prompt】
{prompt_text}

【变异要求】
1. 保持所有格式化占位符不变 ({{resource}}, {{patient}}, {{expert}}, {{reference}})
2. 变异强度: {intensity}
3. 融入【{self.specialty}】的专科特色，强化该领域的诊断思维
4. 可以调整：
   - 诊断策略侧重（更关注{self.expert_template['focus_areas'][0]}等）
   - 推理风格（更严谨/更全面/更关注风险等）
   - 输出结构的细节要求
5. 直接输出修改后的 Prompt 内容，不要任何解释或 Markdown 标记。
"""
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(mutation_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
            except Exception as e:
                print(f"    警告：变异操作失败 (尝试 {attempt+1}/{max_retries}): {e}")
        
        return prompt_text

    def _crossover_prompts(self, prompt_a: str, prompt_b: str, 
                          max_retries: int = 2) -> str:
        """重写：交叉时保持专科特色"""
        crossover_prompt = f"""
你是一个医疗AI Prompt优化专家。请将以下两个【{self.specialty}】诊断 Prompt 进行【交叉融合】。

【该科室专家特征】
- 名称: {self.expert_template['name']}
- 关注领域: {', '.join(self.expert_template['focus_areas'])}
- 思维模式: {', '.join(self.expert_template.get('thinking_patterns', []))}

【父代 Prompt A】
{prompt_a}

【父代 Prompt B】
{prompt_b}

【融合要求】
1. 提取两者最好的指令部分进行组合
2. 必须保留所有格式化占位符 ({{resource}}, {{patient}}, {{expert}}, {{reference}})
3. 确保融合后的Prompt保持【{self.specialty}】的专科特色
4. 生成逻辑通顺、指令清晰的新 Prompt
5. 直接输出新 Prompt 内容，不要任何解释。
"""
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(crossover_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
            except Exception as e:
                print(f"    警告：交叉操作失败 (尝试 {attempt+1}/{max_retries}): {e}")
        
        import random
        return random.choice([prompt_a, prompt_b])

    def run_evolution(self, validation_patients: List[PatientInfo], 
                      generations: int = 10, top_k_return: int = 2) -> List[Dict]:
        """运行进化并返回包含完整专家设定的结果"""
        # 调用父类进化
        raw_results = super().run_evolution(validation_patients, generations, top_k_return)
        
        # 为每个结果添加完整的专家设定
        enriched_results = []
        for i, result in enumerate(raw_results):
            expert_data = {
                # 专家身份信息
                "id": f"{self.expert_template['id']}_{i+1}",
                "name": f"{self.expert_template['name']}（变体{i+1}）",
                "specialty": self.specialty,
                "description": self.expert_template['description'],
                "focus_areas": self.expert_template['focus_areas'],
                "thinking_patterns": self.expert_template.get('thinking_patterns', []),
                # 进化后的Prompt
                "evolved_prompt": result['prompt'],
                # 评估指标
                "fitness": result.get('fitness', 0),
                "stats": result.get('stats', {}),
            }
            enriched_results.append(expert_data)
        
        return enriched_results


def run_specialty_evolution(
    specialty: str,
    patients: List[PatientInfo],
    pipeline: DiagnosticPipeline,
    generations: int = 10,
    population_size: int = 16,
    top_k_return: int = 2
) -> List[Dict]:
    """
    针对单个科室运行GA进化
    
    Args:
        specialty: 科室名称
        patients: 该科室的验证病例
        pipeline: 诊断流水线
        generations: 进化代数
        population_size: 种群大小
        top_k_return: 返回的最优专家数量
    
    Returns:
        该科室的Top K专家列表（包含完整专家设定）
    """
    print(f"\n{'='*70}")
    print(f"🧬 开始进化: 【{specialty}】专家池")
    print(f"  - 验证病例数: {len(patients)}")
    print(f"  - 种群大小: {population_size}")
    print(f"  - 进化代数: {generations}")
    print(f"{'='*70}")
    
    if len(patients) == 0:
        print(f"  ❌ 无有效病例，跳过该科室")
        return []
    
    # 获取该科室的专家模版
    expert_template = EXPERT_TEMPLATES.get(specialty)
    if not expert_template:
        print(f"  ❌ 未找到科室模版: {specialty}")
        return []
    
    print(f"  📋 专家模版: {expert_template['name']}")
    print(f"     关注领域: {', '.join(expert_template['focus_areas'])}")
    print(f"     思维模式: {', '.join(expert_template.get('thinking_patterns', []))}")
    
    # 使用专科定向优化器
    optimizer = SpecialtyPromptOptimizer(
        specialty=specialty,
        expert_template=expert_template,
        base_prompt=system_step3_prompt,
        pipeline=pipeline,
        population_size=population_size,
        elitism_count=max(2, population_size // 8)
    )
    
    # 运行进化
    best_experts = optimizer.run_evolution(
        validation_patients=patients,
        generations=generations,
        top_k_return=top_k_return
    )
    
    print(f"\n✅ 【{specialty}】进化完成，获得 {len(best_experts)} 个优选专家")
    for i, expert in enumerate(best_experts):
        print(f"   [{i+1}] {expert['name']} - Fitness: {expert.get('fitness', 0):.4f}")
    
    return best_experts


def main():
    """主函数：分科训练 + 合并专家池"""
    
    print("="*70)
    print("🏥 分科赛道遗传算法进化系统")
    print("="*70)
    print(f"目标科室: {len(SPECIALTIES)} 个")
    print(f"每科病例: 5 个")
    print(f"每科迭代: 10 代")
    print(f"每科产出: 2 个最优专家")
    print(f"最终专家池: {len(SPECIALTIES) * 2} 人")
    print("="*70)
    
    # 加载数据
    print("\n📂 正在加载患者数据...")
    data_file = "guilin_inpatient_extracted_10000.xlsx"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    df = pd.read_excel(data_file)
    print(f"共加载 {len(df)} 条患者记录")
    
    # 统计各科室病例数
    print("\n📊 各科室病例分布:")
    for specialty in SPECIALTIES:
        aliases = DEPT_ALIASES.get(specialty, [specialty])
        count = df['normalized_name'].apply(
            lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False
        ).sum()
        print(f"  - {specialty}: {count} 例")
    
    # 初始化诊断流水线（共享，节省资源）
    print("\n🔧 正在初始化诊断流水线...")
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",
        enable_rag=False,       # 训练时关闭RAG加速
        enable_experience=False,
        enable_case=False,
    )
    
    # 分科训练
    all_experts = []
    failed_specialties = []
    
    for idx, specialty in enumerate(SPECIALTIES):
        print(f"\n\n{'#'*70}")
        print(f"# 进度: [{idx+1}/{len(SPECIALTIES)}] {specialty}")
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
                population_size=16,  # 小种群节省成本
                top_k_return=2
            )
            all_experts.extend(experts)
            
            # 保存中间结果（防止中途崩溃丢失）
            intermediate_file = f"specialty_pool_{specialty}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(experts, f, ensure_ascii=False, indent=2)
            print(f"  💾 中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
            failed_specialties.append(specialty)
            continue
        
        # 短暂休息，避免API限流
        time.sleep(2)
    
    # 合并最终专家池
    print("\n\n" + "="*70)
    print("🏆 合并最终专家池")
    print("="*70)
    
    # 按fitness排序
    all_experts.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    
    # 保存最终结果
    output_file = "optimized_expert_pool_28.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 最终专家池已保存: {output_file}")
    print(f"   共 {len(all_experts)} 位专家")
    
    # 汇总报告
    print("\n📋 训练汇总:")
    print(f"  - 成功科室: {len(SPECIALTIES) - len(failed_specialties)} 个")
    if failed_specialties:
        print(f"  - 失败科室: {failed_specialties}")
    
    print("\n🎉 专家池分布:")
    specialty_counts = {}
    for expert in all_experts:
        s = expert.get('specialty', '未知')
        specialty_counts[s] = specialty_counts.get(s, 0) + 1
    
    for s, c in specialty_counts.items():
        print(f"  - {s}: {c} 位专家")
    
    # 显示专家详情
    print("\n📋 专家详情:")
    for i, expert in enumerate(all_experts):
        print(f"  [{i+1:2d}] {expert.get('name', '未知')}")
        print(f"       专业: {expert.get('specialty', '未知')}")
        print(f"       关注: {', '.join(expert.get('focus_areas', []))}")
        print(f"       适应度: {expert.get('fitness', 0):.4f}")
    
    # 清理中间文件（可选）
    # for specialty in SPECIALTIES:
    #     intermediate_file = f"specialty_pool_{specialty}.json"
    #     if os.path.exists(intermediate_file):
    #         os.remove(intermediate_file)
    
    return all_experts


if __name__ == "__main__":
    main()

