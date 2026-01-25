"""
分科赛道遗传算法进化脚本 (MOA模式)

针对16个Department分别训练Expert池，每个Department：
1. 从数据集中筛选该Department的5个Case作为验证集
2. 基于该Department的Expert模版进行10代GA进化（在MOA多学科会诊流程中评估）
3. 取出Top 4个最优Expert（包含完整的Expert设定）

最终合并为64人Expert池
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


# 14个Department的完整Expert定义模版
EXPERT_TEMPLATES = {
    "妇产科": {
        "id": "obgyn",
        "name": "妇产科Expert",
        "specialty": "妇产科",
        "description": "擅长女性生殖系统疾病的诊治及围产期保健，关注女性全生命周期健康",
        "focus_areas": ["孕产期管理", "妇科肿瘤", "月经失调", "生殖道炎症"],
        "thinking_patterns": ["月经周期评估", "妊娠相关鉴别", "激素水平分析", "盆腔影像解读"],
    },
    "消化内科": {
        "id": "gastroenterology",
        "name": "消化内科Expert",
        "specialty": "消化内科",
        "description": "擅长食管、胃肠、肝胆胰等消化系统疾病的内科诊治及内镜检查",
        "focus_areas": ["胃炎与溃疡", "肝脏疾病", "功能性胃肠病", "消化道出血"],
        "thinking_patterns": ["腹痛定位分析", "肝功能评估", "内镜指征判断", "幽门螺杆菌筛查"],
    },
    "儿科": {
        "id": "pediatrics",
        "name": "儿科Expert",
        "specialty": "儿科",
        "description": "专注新生儿至青少年时期的生长发育及疾病诊治，关注儿童特有的生理病理特点",
        "focus_areas": ["呼吸道感染", "生长发育评估", "小儿消化系统", "新生儿护理"],
        "thinking_patterns": ["Age相关鉴别", "生长曲线评估", "疫苗接种史", "喂养方式分析"],
    },
    "内分泌科": {
        "id": "endocrinology",
        "name": "内分泌科Expert",
        "specialty": "内分泌科",
        "description": "擅长激素分泌异常及代谢性疾病的Diagnosis与长期管理",
        "focus_areas": ["糖尿病管理", "甲状腺疾病", "骨质疏松", "肥胖与代谢综合征"],
        "thinking_patterns": ["血糖谱分析", "甲功五项解读", "胰岛功能评估", "代谢指标整合"],
    },
    "肝胆外科": {
        "id": "hepatobiliary_surgery",
        "name": "肝胆外科Expert",
        "specialty": "肝胆外科",
        "description": "擅长肝脏、胆道及胰腺疾病的外科手术治疗及围手术期管理",
        "focus_areas": ["胆石症", "肝脏肿瘤", "胰腺炎", "胆道梗阻"],
        "thinking_patterns": ["黄疸鉴别", "肝功能储备评估", "手术指征判断", "影像学分期"],
    },
    "骨科": {
        "id": "orthopedics",
        "name": "骨科Expert",
        "specialty": "骨科",
        "description": "擅长骨骼、关节、肌肉及韧带等运动系统疾病的Diagnosis、复位及手术治疗",
        "focus_areas": ["骨折创伤", "关节炎", "脊柱疾病", "运动损伤"],
        "thinking_patterns": ["受伤机制分析", "X线骨折分型", "关节活动度评估", "神经损伤筛查"],
    },
    "呼吸内科": {
        "id": "respiratory",
        "name": "呼吸内科Expert",
        "specialty": "呼吸内科",
        "description": "擅长呼吸系统感染、气道疾病及肺部肿瘤的内科Diagnosis与治疗",
        "focus_areas": ["慢性阻塞性肺病", "哮喘管理", "肺部结节", "肺部感染"],
        "thinking_patterns": ["肺功能评估", "影像学征象解读", "感染vs肿瘤鉴别", "气道反应性分析"],
    },
    "急诊科": {
        "id": "emergency",
        "name": "急诊科Expert",
        "specialty": "急诊科",
        "description": "擅长急性病、创伤及各类危重症的初步评估、急救复苏与分诊",
        "focus_areas": ["生命体征维持", "急性中毒", "多发伤", "心肺复苏"],
        "thinking_patterns": ["ABCDE评估", "危重症识别", "快速分诊", "时间敏感性判断"],
    },
    "泌尿外科": {
        "id": "urology",
        "name": "泌尿外科Expert",
        "specialty": "泌尿外科",
        "description": "擅长泌尿系统（肾、膀胱、尿路）及男性生殖系统疾病的外科及微创治疗",
        "focus_areas": ["泌尿系结石", "前列腺疾病", "泌尿系肿瘤", "尿路感染"],
        "thinking_patterns": ["尿常规解读", "PSA分析", "结石成分分析", "排尿功能评估"],
    },
    "全科医学科": {
        "id": "general_practice",
        "name": "全科医学Expert",
        "specialty": "全科医学科",
        "description": "提供综合性、连续性的基本医疗服务，擅长未分化疾病的初诊与常见病管理",
        "focus_areas": ["健康查体解读", "常见病初诊", "双向转诊", "慢病长期随访"],
        "thinking_patterns": ["整体评估", "风险分层", "预防医学", "多系统整合"],
    },
    "胃肠外科": {
        "id": "gastro_surgery",
        "name": "胃肠外科Expert",
        "specialty": "胃肠外科",
        "description": "擅长胃、小肠、结直肠及肛门疾病的手术治疗，特别是肿瘤及急腹症处理",
        "focus_areas": ["胃肠道肿瘤", "阑尾炎", "肠梗阻", "疝气修补"],
        "thinking_patterns": ["急腹症鉴别", "肠梗阻分型", "肿瘤分期", "手术时机判断"],
    },
    "胸心血管外科": {
        "id": "cardiothoracic_surgery",
        "name": "胸心外科Expert",
        "specialty": "胸心血管外科",
        "description": "擅长胸腔内器官（心脏、大血管、肺、食管）的复杂外科手术治疗",
        "focus_areas": ["肺癌手术", "心脏瓣膜病", "冠脉搭桥", "主动脉夹层"],
        "thinking_patterns": ["胸片CT解读", "心功能评估", "手术风险分层", "术前优化"],
    },
    "肿瘤科": {
        "id": "oncology",
        "name": "肿瘤科Expert",
        "specialty": "肿瘤科",
        "description": "擅长各类良恶性肿瘤的内科综合治疗，包括化疗、靶向治疗及免疫治疗",
        "focus_areas": ["放化疗方案", "肿瘤筛查", "癌痛管理", "多学科会诊(MDT)"],
        "thinking_patterns": ["肿瘤标志物解读", "TNM分期", "基因突变分析", "预后评估"],
    },
    "心血管内科": {
        "id": "cardiology",
        "name": "心血管内科Expert",
        "specialty": "心血管内科",
        "description": "擅长心脏及血管疾病的内科诊疗与介入治疗，关注心血管风险防控",
        "focus_areas": ["高血压", "冠心病", "心律失常", "心力衰竭"],
        "thinking_patterns": ["心电图分析", "心肌酶谱解读", "危险分层", "抗凝策略"],
    },
    "风湿免疫科": {
        "id": "rheumatology",
        "name": "风湿免疫科Expert",
        "specialty": "风湿免疫科",
        "description": "擅长各类风湿性疾病及自身免疫性疾病的Diagnosis与长期慢病管理",
        "focus_areas": ["类风湿关节炎", "系统性红斑狼疮", "强直性脊柱炎", "痛风"],
        "thinking_patterns": ["自身抗体谱分析", "多系统受累评估", "免疫调节策略", "炎症指标监测"],
    },
    "神经内科": {
        "id": "neurology",
        "name": "神经内科Expert",
        "specialty": "神经内科",
        "description": "擅长中枢神经系统、周围神经系统及骨骼肌疾病的内科Diagnosis与治疗",
        "focus_areas": ["脑血管病", "癫痫", "帕金森病", "周围神经病"],
        "thinking_patterns": ["神经系统查体", "定位定性Diagnosis", "脑影像分析", "认知功能评估"],
    },
}

# 16个Department列表
SPECIALTIES = list(EXPERT_TEMPLATES.keys())

# Department名称映射（处理数据中可能的别名）
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
    "风湿免疫科": ["风湿免疫科", "风湿科", "免疫科"],
    "神经内科": ["神经内科", "神内", "脑病科"],
}


def load_patients_by_specialty(df: pd.DataFrame, specialty: str, count: int = 5) -> List[PatientInfo]:
    """
    从DataFrame中筛选指定Department的Patient
    
    Args:
        df: Patient数据DataFrame
        specialty: Department名称
        count: 需要的Case数量
    
    Returns:
        该Department的Patient列表
    """
    aliases = DEPT_ALIASES.get(specialty, [specialty])
    
    # 筛选该Department的有效Case
    valid_mask = df['is_history_cleaned'] == True
    dept_mask = df['normalized_name'].apply(lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False)
    
    filtered_df = df[valid_mask & dept_mask]
    
    if len(filtered_df) == 0:
        print(f"  ⚠️ Department '{specialty}' 未找到有效Case，尝试模糊匹配...")
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
            print(f"    解析Patient失败: {e}")
            continue
    
    return patients


class SpecialtyPromptOptimizer(GeneticPromptOptimizer):
    """
    专科定向Prompt优化器
    
    继承自通用优化器，但：
    1. 使用该Department的Expert模版进行评估
    2. 产出包含完整Expert设定的结构化数据
    """
    
    def __init__(self, specialty: str, expert_template: Dict, 
                 base_prompt: str, pipeline: DiagnosticPipeline,
                 population_size: int = 16, elitism_count: int = 2):
        super().__init__(base_prompt, pipeline, population_size, elitism_count)
        self.specialty = specialty
        self.expert_template = expert_template
        
    def _get_matched_expert(self, patient: PatientInfo) -> Dict:
        """重写：始终返回该Department的Expert模版"""
        return self.expert_template
    
    def _mutate_prompt(self, prompt_text: str, intensity: str = "medium", 
                      max_retries: int = 2) -> str:
        """重写：变异时融入专科特色"""
        mutation_prompt = f"""
你是一个医疗AI Prompt优化Expert。请对以下用于【{self.specialty}】Diagnosis的 System Prompt 进行【变异】操作。

【该DepartmentExpert特征】
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
3. 融入【{self.specialty}】的专科特色，强化该领域的Diagnosis思维
4. 可以调整：
   - Diagnosis策略侧重（更关注{self.expert_template['focus_areas'][0]}等）
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
你是一个医疗AI Prompt优化Expert。请将以下两个【{self.specialty}】Diagnosis Prompt 进行【交叉融合】。

【该DepartmentExpert特征】
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

    def evaluate_fitness(self, validation_patients: List[PatientInfo], 
                         sample_size: int = 5) -> None:
        """
        重写：在MOA多学科会诊流程中评估Expert适应度
        
        不再单独评估单个Expert的Diagnosis准确性，而是评估：
        该Expert(使用进化Prompt)参与多学科会诊后，对最终Step-4整合结论的贡献
        """
        print(f"\n开始评估第 {self.generation} 代适应度 (样本数: {sample_size}) [MOA模式]...")
        
        # 1. 筛选验证集 (同父类逻辑)
        eval_batch = validation_patients[:sample_size]
        
        # 2. 预计算每个Case的"会诊上下文" (Context)
        # 即：除了当前正在优化的专科外，其他Expert对该Case的意见
        # 这部分对于同一代的所有个体是共用的，只需计算一次
        print("  - 预计算会诊上下文 (Step 1 & Step 2/3 for other experts)...")
        context_cache = {} # {patient_id: {'other_opinions': [], 'step1': ...}}
        
        for patient in eval_batch:
            try:
                # 运行 Step 1 路由
                step1_res = self.pipeline.step1_route(patient)
                
                # 获取推荐Expert列表
                recommended_experts = self.pipeline._activate_experts_step1(step1_res["output"])
                
                # 确保当前优化的专科在列表中 (如果不在，强制加入)
                # 注意：这里我们要找的是"对手"或"搭档"，所以要区分开
                other_experts = []
                current_specialty_expert = None
                
                for exp in recommended_experts:
                    # 简单判断：如果ExpertID或名称包含当前专科关键词
                    if self.expert_template['id'] == exp['id']:
                        current_specialty_expert = exp
                    else:
                        other_experts.append(exp)
                
                # 如果Step1没推荐当前专科，我们需要强制让它参与，否则无法评估
                if not current_specialty_expert:
                    current_specialty_expert = self.expert_template
                
                # 运行其他Expert的 Step 2 & 3 (作为固定背景)
                other_opinions = []
                for other_exp in other_experts:
                    # Step 2
                    s2 = self.pipeline.step2_semantic_rewrite(patient, other_exp)
                    # Step 3 (使用默认Prompt)
                    s3 = self.pipeline.step3_diagnosis(patient, other_exp, s2["output"], auto_retrieve=False)
                    
                    other_opinions.append({
                        "expert_id": other_exp['id'],
                        "expert_name": other_exp['name'],
                        "specialty": other_exp['specialty'],
                        "rewrite": s2["output"],
                        "diagnosis": s3["output"],
                        "output": s3["output"]
                    })
                
                context_cache[patient.patient_id] = {
                    "other_opinions": other_opinions,
                    "target_expert_base": current_specialty_expert
                }
                
            except Exception as e:
                print(f"    警告：预计算上下文失败 {patient.patient_id}: {e}")
                continue
                
        # 3. 评估每个个体 (Prompt变体)
        for idx, individual in enumerate(self.population):
            if individual['fitness'] > 0:
                continue
                
            print(f"  - 评估个体 {idx+1}/{self.population_size} (ID: {individual['id']})...")
            
            total_score = 0
            correct_count = 0
            valid_evaluations = 0
            
            for patient in eval_batch:
                if patient.patient_id not in context_cache:
                    continue
                    
                context = context_cache[patient.patient_id]
                target_expert = context["target_expert_base"]
                
                try:
                    # === 运行当前个体的 Step 2 & 3 ===
                    # Step 2 (语义重写) - 可以复用缓存或重新运行，这里简化为重新运行
                    step2_res = self.pipeline.step2_semantic_rewrite(patient, target_expert)
                    
                    # Step 3 (使用进化的Prompt)
                    step3_res = self.pipeline.step3_diagnosis(
                        patient, target_expert, 
                        step2_res["output"], 
                        custom_prompt_template=individual['prompt'],
                        auto_retrieve=False
                    )
                    
                    # 构建当前Expert的意见记录
                    current_opinion = {
                        "expert_id": target_expert['id'],
                        "expert_name": target_expert['name'],
                        "specialty": target_expert['specialty'],
                        "rewrite": step2_res["output"],
                        "diagnosis": step3_res["output"],
                        "output": step3_res["output"]
                    }
                    
                    # === 组合所有Expert意见 ===
                    # 将当前Expert的意见加入到"其他Expert"中
                    all_opinions = context["other_opinions"] + [current_opinion]
                    
                    # === 运行 Step 4 (MOA 聚合) ===
                    step4_res = self.pipeline.step4_aggregate(patient, all_opinions)
                    
                    # === 评估最终结果 ===
                    # 这里的关键是：我们评估的是 Step 4 的综合结论，而非单Expert结论
                    score, is_correct = self._judge_diagnosis(patient, step4_res['output'])
                    
                    total_score += score
                    if is_correct:
                        correct_count += 1
                    valid_evaluations += 1
                    
                except Exception as e:
                    print(f"    警告：评估失败 {patient.patient_id}: {e}")
                    continue
            
            if valid_evaluations == 0:
                individual['fitness'] = 0.0
                individual['stats'] = {'accuracy': 0.0, 'avg_score': 0.0, 'valid_count': 0}
                continue
            
            # 计算平均分
            avg_score = total_score / valid_evaluations
            accuracy = correct_count / valid_evaluations
            
            # 适应度公式
            fitness = accuracy * 0.7 + (avg_score / 100) * 0.3
            
            individual['fitness'] = fitness
            individual['stats'] = {
                'accuracy': accuracy, 
                'avg_score': avg_score, 
                'valid_count': valid_evaluations
            }
            print(f"    -> Fitness: {fitness:.4f} (MOA Acc: {accuracy:.0%}, Avg: {avg_score:.1f})")

    def run_evolution(self, validation_patients: List[PatientInfo], 
                      generations: int = 10, top_k_return: int = 2) -> List[Dict]:
        """运行进化并返回包含完整Expert设定的结果"""
        # 调用父类进化
        raw_results = super().run_evolution(validation_patients, generations, top_k_return)
        
        # 为每个结果添加完整的Expert设定
        enriched_results = []
        for i, result in enumerate(raw_results):
            expert_data = {
                # Expert身份信息
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
    针对单个Department运行GA进化
    
    Args:
        specialty: Department名称
        patients: 该Department的验证Case
        pipeline: Diagnosis流水线
        generations: 进化代数
        population_size: 种群大小
        top_k_return: 返回的最优Expert数量
    
    Returns:
        该Department的Top KExpert列表（包含完整Expert设定）
    """
    print(f"\n{'='*70}")
    print(f"🧬 开始进化: 【{specialty}】Expert池")
    print(f"  - 验证Case数: {len(patients)}")
    print(f"  - 种群大小: {population_size}")
    print(f"  - 进化代数: {generations}")
    print(f"{'='*70}")
    
    if len(patients) == 0:
        print(f"  ❌ 无有效Case，跳过该Department")
        return []
    
    # 获取该Department的Expert模版
    expert_template = EXPERT_TEMPLATES.get(specialty)
    if not expert_template:
        print(f"  ❌ 未找到Department模版: {specialty}")
        return []
    
    print(f"  📋 Expert模版: {expert_template['name']}")
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
    
    print(f"\n✅ 【{specialty}】进化完成，获得 {len(best_experts)} 个优选Expert")
    for i, expert in enumerate(best_experts):
        print(f"   [{i+1}] {expert['name']} - Fitness: {expert.get('fitness', 0):.4f}")
    
    return best_experts


def main():
    """主函数：分科训练 + 合并Expert池"""
    
    print("="*70)
    print("🏥 分科赛道遗传算法进化系统")
    print("="*70)
    print(f"目标Department: {len(SPECIALTIES)} 个")
    print(f"每科Case: 5 个")
    print(f"每科迭代: 10 代")
    print(f"每科产出: 4 个最优Expert")
    print(f"最终Expert池: {len(SPECIALTIES) * 4} 人")
    print("="*70)
    
    # 加载数据
    print("\n📂 正在加载Patient数据...")
    data_file = "guilin_inpatient_extracted_10000(2).xlsx"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    df = pd.read_excel(data_file)
    print(f"共加载 {len(df)} 条Patient记录")
    
    # 统计各DepartmentCase数
    print("\n📊 各DepartmentCase分布:")
    for specialty in SPECIALTIES:
        aliases = DEPT_ALIASES.get(specialty, [specialty])
        count = df['normalized_name'].apply(
            lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False
        ).sum()
        print(f"  - {specialty}: {count} 例")
    
    # 初始化Diagnosis流水线（共享，节省资源）
    print("\n🔧 正在初始化Diagnosis流水线...")
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
        
        # 获取该Department的验证Case
        patients = load_patients_by_specialty(df, specialty, count=5)
        print(f"  筛选到 {len(patients)} 个Case")
        
        if len(patients) < 3:
            print(f"  ⚠️ Case数不足3个，跳过该Department")
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
                top_k_return=4
            )
            all_experts.extend(experts)
            
            # 保存中间结果（防止中途崩溃丢失）
            intermediate_file = f"moa_specialty_pool_{specialty}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(experts, f, ensure_ascii=False, indent=2)
            print(f"  💾 中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
            failed_specialties.append(specialty)
            continue
        
        # 短暂休息，避免API限流
        time.sleep(2)
    
    # 合并最终Expert池
    print("\n\n" + "="*70)
    print("🏆 合并最终Expert池")
    print("="*70)
    
    # 按fitness排序
    all_experts.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    
    # 保存最终结果
    output_file = "moa_optimized_expert_pool_64.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 最终Expert池已保存: {output_file}")
    print(f"   共 {len(all_experts)} 位Expert")
    
    # 汇总报告
    print("\n📋 训练汇总:")
    print(f"  - 成功Department: {len(SPECIALTIES) - len(failed_specialties)} 个")
    if failed_specialties:
        print(f"  - 失败Department: {failed_specialties}")
    
    print("\n🎉 Expert池分布:")
    specialty_counts = {}
    for expert in all_experts:
        s = expert.get('specialty', '未知')
        specialty_counts[s] = specialty_counts.get(s, 0) + 1
    
    for s, c in specialty_counts.items():
        print(f"  - {s}: {c} 位Expert")
    
    # 显示Expert详情
    print("\n📋 Expert详情:")
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

