"""
混合多学科专家智能体诊断系统

支持三种专家激活模式：
1. Step-1路由模式: 基于LLM会诊建议匹配专家
2. EEP语义激活模式: 基于Episode语义相似性激活专家
3. Evolved Pool模式: 使用遗传算法进化后的最优专家池
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
import time

from system_step1_route import system_step1_prompt
from system_step2_ir import system_step2_prompt
from system_step3_diag import system_step3_prompt
from system_step4_agg import system_step4_prompt

# 导入可演化专家资源池
from expert_pool import EvolvingExpertPool, DiagnosticEpisode, ExpertUnit

# 导入知识检索服务
from knowledge_retriever import KnowledgeRetriever

API_BASE_URL = "https://yunwu.ai/v1"
API_KEY = "sk-mZ1tJ8giPu2WqauY5SivguiTVJmFolWNAkBQ4i5Y3Lh2jxVL"
MODEL_NAME = "gpt-4o"

EXPERTS_CONFIG = [
    {
        "id": "obgyn",
        "name": "妇产科专家",
        "specialty": "妇产科",
        "description": "擅长女性生殖系统疾病的诊治及围产期保健，关注女性全生命周期健康",
        "focus_areas": ["孕产期管理", "妇科肿瘤", "月经失调", "生殖道炎症"],
    },
    {
        "id": "gastroenterology",
        "name": "消化内科专家",
        "specialty": "消化内科",
        "description": "擅长食管、胃肠、肝胆胰等消化系统疾病的内科诊治及内镜检查",
        "focus_areas": ["胃炎与溃疡", "肝脏疾病", "功能性胃肠病", "消化道出血"],
    },
    {
        "id": "pediatrics",
        "name": "儿科专家",
        "specialty": "儿科",
        "description": "专注新生儿至青少年时期的生长发育及疾病诊治，关注儿童特有的生理病理特点",
        "focus_areas": ["呼吸道感染", "生长发育评估", "小儿消化系统", "新生儿护理"],
    },
    {
        "id": "endocrinology",
        "name": "内分泌科专家",
        "specialty": "内分泌科",
        "description": "擅长激素分泌异常及代谢性疾病的诊断与长期管理",
        "focus_areas": ["糖尿病管理", "甲状腺疾病", "骨质疏松", "肥胖与代谢综合征"],
    },
    {
        "id": "hepatobiliary_surgery",
        "name": "肝胆外科专家",
        "specialty": "肝胆外科",
        "description": "擅长肝脏、胆道及胰腺疾病的外科手术治疗及围手术期管理",
        "focus_areas": ["胆石症", "肝脏肿瘤", "胰腺炎", "胆道梗阻"],
    },
    {
        "id": "orthopedics",
        "name": "骨科专家",
        "specialty": "骨科",
        "description": "擅长骨骼、关节、肌肉及韧带等运动系统疾病的诊断、复位及手术治疗",
        "focus_areas": ["骨折创伤", "关节炎", "脊柱疾病", "运动损伤"],
    },
    {
        "id": "respiratory",
        "name": "呼吸内科专家",
        "specialty": "呼吸内科",
        "description": "擅长呼吸系统感染、气道疾病及肺部肿瘤的内科诊断与治疗",
        "focus_areas": ["慢性阻塞性肺病", "哮喘管理", "肺部结节", "肺部感染"],
    },
    {
        "id": "emergency",
        "name": "急诊科专家",
        "specialty": "急诊科",
        "description": "擅长急性病、创伤及各类危重症的初步评估、急救复苏与分诊",
        "focus_areas": ["生命体征维持", "急性中毒", "多发伤", "心肺复苏"],
    },
    {
        "id": "urology",
        "name": "泌尿外科专家",
        "specialty": "泌尿外科",
        "description": "擅长泌尿系统（肾、膀胱、尿路）及男性生殖系统疾病的外科及微创治疗",
        "focus_areas": ["泌尿系结石", "前列腺疾病", "泌尿系肿瘤", "尿路感染"],
    },
    {
        "id": "general_practice",
        "name": "全科医学专家",
        "specialty": "全科医学科",
        "description": "提供综合性、连续性的基本医疗服务，擅长未分化疾病的初诊与常见病管理",
        "focus_areas": ["健康查体解读", "常见病初诊", "双向转诊", "慢病长期随访"],
    },
    {
        "id": "gastro_surgery",
        "name": "胃肠外科专家",
        "specialty": "胃肠外科",
        "description": "擅长胃、小肠、结直肠及肛门疾病的手术治疗，特别是肿瘤及急腹症处理",
        "focus_areas": ["胃肠道肿瘤", "阑尾炎", "肠梗阻", "疝气修补"],
    },
    {
        "id": "cardiothoracic_surgery",
        "name": "胸心外科专家",
        "specialty": "胸心血管外科",
        "description": "擅长胸腔内器官（心脏、大血管、肺、食管）的复杂外科手术治疗",
        "focus_areas": ["肺癌手术", "心脏瓣膜病", "冠脉搭桥", "主动脉夹层"],
    },
    {
        "id": "oncology",
        "name": "肿瘤科专家",
        "specialty": "肿瘤科",
        "description": "擅长各类良恶性肿瘤的内科综合治疗，包括化疗、靶向治疗及免疫治疗",
        "focus_areas": ["放化疗方案", "肿瘤筛查", "癌痛管理", "多学科会诊(MDT)"],
    },
    {
        "id": "cardiology",
        "name": "心血管内科专家",
        "specialty": "心血管内科",
        "description": "擅长心脏及血管疾病的内科诊疗与介入治疗，关注心血管风险防控",
        "focus_areas": ["高血压", "冠心病", "心律失常", "心力衰竭"],
    },
    {
        "id": "rheumatology",
        "name": "风湿免疫科专家",
        "specialty": "风湿免疫科",
        "description": "擅长各类风湿性疾病及自身免疫性疾病的诊断与长期慢病管理",
        "focus_areas": ["类风湿关节炎", "系统性红斑狼疮", "强直性脊柱炎", "痛风"],
    },
    {
        "id": "neurology",
        "name": "神经内科专家",
        "specialty": "神经内科",
        "description": "擅长中枢神经系统、周围神经系统及骨骼肌疾病的内科诊断与治疗",
        "focus_areas": ["脑血管病", "癫痫", "帕金森病", "周围神经病"],
    }
]


@dataclass
class PatientInfo:
    """患者信息数据类"""
    patient_id: str
    gender: str
    age: int
    department: str
    chief_complaint: str
    history_of_present_illness: str
    past_history: str
    personal_history: str
    physical_examination: str
    labs: str
    imaging: str
    main_diagnosis: str
    main_diagnosis_icd: str
    
    def to_prompt_string(self) -> str:
        """将患者信息格式化为prompt字符串"""
        return f"""
患者ID: {self.patient_id}
性别: {self.gender}
年龄: {self.age}岁
就诊科室: {self.department}

【主诉】
{self.chief_complaint}

【现病史】
{self.history_of_present_illness}

【既往史】
{self.past_history}

【个人史】
{self.personal_history}

【体格检查】
{self.physical_examination}

【实验室检查】
{self.labs}

【影像学检查】
{self.imaging}
"""


class DiagnosticPipeline:
    """
    多学科诊断流水线
    
    支持三种专家激活模式：
    - 'step1_route': 基于Step-1会诊建议的专科匹配
    - 'eep_semantic': 基于EEP语义相似性激活（推荐）
    - 'evolved_pool': 使用遗传算法进化后的最优专家池
    
    支持三大知识来源：
    - RAG医学指南检索
    - 经验库检索 (A-Mem)
    - 病例库检索 (ACE)
    """
    
    def __init__(
        self, 
        activation_mode: str = "eep_semantic", 
        expert_pool_path: str = "outputs/expert_pool.json",
        evolved_pool_path: str = "outputs/moa_optimized_expert_pool_64.json",
        enable_rag: bool = True,
        enable_experience: bool = True,
        enable_case: bool = True,
        rag_index_dir: str = "rag/rag_index",
        memory_db_root: str = "exp/A-mem-sys/A-mem-sys/memory_db",
        experience_collection: str = "experience_100000",
        case_collection: str = "case_100000",
    ):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
        self.activation_mode = activation_mode
        self.resource = """【医疗资源】
可用科室：内科、外科、急诊科、心内科、消化内科、呼吸内科、神经内科、肿瘤科、骨科、泌尿外科、肝胆外科、妇产科、儿科等
知识库：医学指南库、临床经验库、相似病例库
说明：本系统仅提供诊断辅助，不替代专业医疗行为。"""
        
        # 初始化可演化专家资源池 (EEP)
        if activation_mode == "eep_semantic":
            print("正在初始化可演化专家资源池 (EEP)...")
            self.expert_pool = EvolvingExpertPool(expert_pool_path)
            self.experts = EXPERTS_CONFIG  # 保留兼容性
        else:
            self.expert_pool = None
            self.experts = EXPERTS_CONFIG
        
        # 初始化进化后的专家池 (GA Evolved Pool)
        self.evolved_pool = None
        self.evolved_pool_path = evolved_pool_path
        if activation_mode == "evolved_pool":
            print("正在加载遗传算法进化后的专家池...")
            self.evolved_pool = self._load_evolved_pool(evolved_pool_path)
            if self.evolved_pool:
                print(f"  ✅ 成功加载 {len(self.evolved_pool)} 个进化专家Prompt")
            else:
                print("  ⚠️ 未找到进化专家池，将使用默认模式")
                self.activation_mode = "step1_route"
        
        # 初始化知识检索服务
        print("正在初始化知识检索服务...")
        self.knowledge_retriever = KnowledgeRetriever(
            enable_rag=enable_rag,
            enable_experience=enable_experience,
            enable_case=enable_case,
            rag_index_dir=rag_index_dir,
            memory_db_root=memory_db_root,
            experience_collection=experience_collection,
            case_collection=case_collection,
            api_key=API_KEY,
            api_base_url=API_BASE_URL,
        )
        
        # 构建专科名称到专家配置的映射（支持多种别名）
        self.specialty_mapping = self._build_specialty_mapping()
    
    def _load_evolved_pool(self, path: str) -> Optional[List[Dict]]:
        """加载遗传算法进化后的专家池"""
        import os
        if not os.path.exists(path):
            print(f"  警告: 进化专家池文件不存在: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            
            # 验证格式
            if not isinstance(pool, list) or len(pool) == 0:
                print(f"  警告: 进化专家池格式无效")
                return None
            
            # 确保每个专家都有必要字段
            for i, expert in enumerate(pool):
                if 'prompt' not in expert:
                    print(f"  警告: 专家{i}缺少prompt字段")
                    return None
                # 添加ID（如果没有）
                if 'id' not in expert:
                    expert['id'] = f"evolved_expert_{i}"
                # 添加名称（如果没有）
                if 'name' not in expert:
                    expert['name'] = f"进化专家-{i+1}"
                    
            return pool
            
        except Exception as e:
            print(f"  警告: 加载进化专家池失败: {e}")
            return None
    
    def _build_specialty_mapping(self) -> Dict[str, Dict]:
        """构建专科名称映射表，支持多种别名"""
        mapping = {}
        
        # 定义别名映射
        aliases = {
            "妇产科": ["妇产科", "妇科", "产科", "妇产"],
            "消化内科": ["消化内科", "消化科", "胃肠内科"],
            "儿科": ["儿科", "小儿科", "儿内科"],
            "内分泌科": ["内分泌科", "内分泌", "代谢科"],
            "肝胆外科": ["肝胆外科", "肝胆胰外科", "肝外科"],
            "骨科": ["骨科", "骨外科", "创伤骨科"],
            "呼吸内科": ["呼吸内科", "呼吸科", "肺科", "呼吸"],
            "急诊科": ["急诊科", "急诊", "急诊医学科"],
            "泌尿外科": ["泌尿外科", "泌尿科", "肾内科", "肾科"],
            "全科医学科": ["全科医学科", "全科", "全科医学", "综合内科"],
            "胃肠外科": ["胃肠外科", "普外科", "胃肠道外科", "结直肠外科"],
            "胸心血管外科": ["胸心血管外科", "胸外科", "心外科", "心胸外科"],
            "肿瘤科": ["肿瘤科", "肿瘤内科", "肿瘤外科", "放疗科"],
            "心血管内科": ["心血管内科", "心内科", "心脏科", "心脏内科", "循环内科"],
        }
        
        for expert in self.experts:
            specialty = expert['specialty']
            mapping[specialty] = expert
            if specialty in aliases:
                for alias in aliases[specialty]:
                    mapping[alias] = expert
        
        return mapping
    
    def _extract_recommended_specialties(self, step1_output: str) -> List[str]:
        """从Step-1输出中提取推荐的专科列表"""
        extract_prompt = f"""请从以下会诊/转诊规划建议中，提取所有推荐的专科/科室名称。

【Step-1输出】
{step1_output}

【可用专科列表】
{', '.join([e['specialty'] for e in self.experts])}

请按照推荐优先级顺序，输出专科名称列表。
仅输出JSON数组格式，不要任何解释。例如：["胃肠外科", "心血管内科", "呼吸内科"]
"""
        
        result = self._call_llm(extract_prompt)
        
        try:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1]
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0]
            result = result.strip()
            
            specialties = json.loads(result)
            if isinstance(specialties, list):
                return specialties
        except (json.JSONDecodeError, TypeError):
            pass
        
        return self._fallback_extract_specialties(step1_output)
    
    def _fallback_extract_specialties(self, text: str) -> List[str]:
        """备用方案：通过关键词匹配提取专科"""
        found = []
        for specialty in self.specialty_mapping.keys():
            if specialty in text and specialty not in found:
                expert = self.specialty_mapping[specialty]
                if expert['specialty'] not in found:
                    found.append(expert['specialty'])
        return found
    
    def _match_experts_by_specialties(self, specialties: List[str]) -> List[Dict]:
        """根据专科名称列表匹配专家配置"""
        matched_experts = []
        matched_ids = set()
        
        for specialty in specialties:
            # 尝试直接匹配
            if specialty in self.specialty_mapping:
                expert = self.specialty_mapping[specialty]
                if expert['id'] not in matched_ids:
                    matched_experts.append(expert)
                    matched_ids.add(expert['id'])
            else:
                # 尝试模糊匹配
                for key, expert in self.specialty_mapping.items():
                    if key in specialty or specialty in key:
                        if expert['id'] not in matched_ids:
                            matched_experts.append(expert)
                            matched_ids.add(expert['id'])
                            break
        
        return matched_experts
    
    def _call_llm(self, system_prompt: str, max_retries: int = 3) -> str:
        """调用LLM API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个专业的医疗诊断辅助智能体。"},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def step1_route(self, patient: PatientInfo) -> Dict[str, Any]:
        """
        Step-1: 会诊/转诊规划
        输入: 医疗资源 r, 患者信息 p
        输出: 专科优先级列表、紧急程度、高风险需排除清单
        """
        print("\n" + "="*60)
        print("【Step-1】会诊/转诊规划")
        print("="*60)
        
        prompt = system_step1_prompt.format(
            resource=self.resource,
            patient=patient.to_prompt_string()
        )
        
        result = self._call_llm(prompt)
        print(result)
        
        return {
            "step": "step1_route",
            "output": result
        }
    
    def step2_semantic_rewrite(self, patient: PatientInfo, expert: Dict) -> Dict[str, Any]:
        """
        Step-2: 专家语义重写
        输入: 医疗资源 r, 患者信息 p, 专家配置 e
        输出: 医学化重写段、结构化检索要素摘要
        """
        expert_info = f"""
专家角色: {expert['name']}
专业领域: {expert['specialty']}
专业描述: {expert['description']}
关注重点: {', '.join(expert['focus_areas'])}
"""
        
        prompt = system_step2_prompt.format(
            resource=self.resource,
            patient=patient.to_prompt_string(),
            expert=expert_info
        )
        
        result = self._call_llm(prompt)
        
        return {
            "step": "step2_semantic_rewrite",
            "expert_id": expert['id'],
            "expert_name": expert['name'],
            "output": result
        }
    
    def step3_diagnosis(self, patient: PatientInfo, expert: Dict, 
                        rewritten_info: str, reference: str = "",
                        auto_retrieve: bool = True,
                        custom_prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Step-3: 专家鉴别诊断
        输入: 医疗资源 r, 患者信息 p, 专家配置 e, 参考资料 ref
        输出: 鉴别诊断列表、风险与警示信号、下一步检查方向
        
        Args:
            patient: 患者信息
            expert: 专家配置
            rewritten_info: Step-2语义重写结果
            reference: 外部提供的参考资料（可选）
            auto_retrieve: 是否自动检索参考资料（默认True）
            custom_prompt_template: 自定义Prompt模版（用于遗传算法进化），若为None则使用默认模版
        """
        expert_info = f"""
专家角色: {expert['name']}
专业领域: {expert['specialty']}
专业描述: {expert['description']}
关注重点: {', '.join(expert['focus_areas'])}

【专家视角下的患者信息重写】
{rewritten_info}
"""
        
        # 参考资料检索
        if not reference and auto_retrieve:
            # 使用专家语义重写的内容进行检索
            reference = self.knowledge_retriever.retrieve_for_expert(
                rewritten_query=rewritten_info,
                expert_specialty=expert['specialty'],
                rag_k=3,
                experience_k=3,
                case_k=3
            )
        elif not reference:
            reference = "【暂无额外参考资料】"
        
        # 使用自定义模版或默认模版
        template = custom_prompt_template if custom_prompt_template else system_step3_prompt
        
        prompt = template.format(
            resource=self.resource,
            patient=patient.to_prompt_string(),
            expert=expert_info,
            reference=reference
        )
        
        result = self._call_llm(prompt)
        
        return {
            "step": "step3_diagnosis",
            "expert_id": expert['id'],
            "expert_name": expert['name'],
            "output": result,
            "reference_used": reference  # 记录使用的参考资料
        }
    
    def step4_aggregate(self, patient: PatientInfo, 
                        expert_opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step-4: 多专家整合裁决
        输入: 医疗资源 r, 患者信息 p, 专家意见集合
        输出: 综合诊断结论、分歧说明、综合风险评估、下一步行动建议
        """
        print("\n" + "="*60)
        print("【Step-4】多专家整合裁决")
        print("="*60)
        
        # 整合所有专家意见
        experts_summary = ""
        for opinion in expert_opinions:
            experts_summary += f"""
{'='*40}
【{opinion['expert_name']}】的诊断意见
{'='*40}
{opinion['output']}

"""
        
        prompt = system_step4_prompt.format(
            resource=self.resource,
            patient=patient.to_prompt_string(),
            experts=experts_summary
        )
        
        result = self._call_llm(prompt)
        print(result)
        
        return {
            "step": "step4_aggregate",
            "output": result
        }
    
    def _expert_unit_to_dict(self, expert_unit: ExpertUnit) -> Dict:
        """将ExpertUnit转换为兼容的字典格式"""
        return {
            'id': expert_unit.id,
            'name': expert_unit.name,
            'specialty': expert_unit.specialty,
            'description': expert_unit.description,
            'focus_areas': expert_unit.focus_areas,
            'thinking_patterns': getattr(expert_unit, 'thinking_patterns', []),
            'risk_preference': expert_unit.metadata.risk_preference if expert_unit.metadata else '中立'
        }
    
    def _activate_experts_eep(
        self, 
        patient: PatientInfo, 
        top_k: int = 5,
        step1_output: str = "",
        step1_weight: float = 0.3
    ) -> Tuple[List[Dict], DiagnosticEpisode]:
        """
        使用EEP混合激活模式选择专家
        
        结合两种信号：
        1. EEP语义激活 (1 - step1_weight): 基于Episode语义相似性
        2. Step-1路由推荐 (step1_weight): 基于LLM会诊建议
        
        Args:
            patient: 患者信息
            top_k: 激活专家数量
            step1_output: Step-1的输出结果
            step1_weight: Step-1推荐的权重 (0-1)，默认0.3
        """
        print("\n" + "-"*60)
        print(f"【EEP混合激活】语义激活 + Step-1路由 (权重: {step1_weight:.0%})")
        print("-"*60)
        
        # 创建诊断Episode
        episode = DiagnosticEpisode.from_patient_info(patient)
        
        print(f"Episode ID: {episode.episode_id}")
        print(f"关键词: {episode.keywords}")
        print(f"系统域: {episode.tags.get('system_domain', [])}")
        print(f"人群特征: {episode.tags.get('population', [])}")
        
        # 从Step-1提取推荐专科
        step1_recommended = []
        if step1_output:
            step1_recommended = self._extract_recommended_specialties(step1_output)
            print(f"Step-1推荐专科: {step1_recommended}")
        
        # 使用EEP激活专家（传入Step-1推荐以计算混合得分）
        activated_units = self.expert_pool.activate_experts_hybrid(
            episode, 
            step1_recommended=step1_recommended,
            step1_weight=step1_weight,
            top_k=top_k
        )
        
        # 转换为兼容格式
        selected_experts = [self._expert_unit_to_dict(unit) for unit in activated_units]
        
        return selected_experts, episode
    
    def _activate_experts_step1(self, step1_output: str) -> List[Dict]:
        """
        使用Step-1路由模式选择专家
        
        基于LLM会诊建议提取专科并匹配专家
        """
        print("\n" + "-"*60)
        print("【Step-1路由】根据会诊建议匹配专家...")
        print("-"*60)
        
        recommended_specialties = self._extract_recommended_specialties(step1_output)
        selected_experts = self._match_experts_by_specialties(recommended_specialties)
        
        print(f"Step-1推荐专科: {recommended_specialties}")
        print(f"匹配到专家: {[e['name'] for e in selected_experts]}")
        
        # 如果没有匹配到任何专家，使用默认专家（全科）
        if not selected_experts:
            print("警告: 未匹配到推荐专家，使用全科医学专家")
            for expert in self.experts:
                if expert['id'] == 'general_practice':
                    selected_experts = [expert]
                    break
        
        return selected_experts
    
    def _activate_experts_evolved(
        self, 
        patient: PatientInfo,
        step1_output: str = "",
        top_k: int = 5,
        selection_strategy: str = "hybrid"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        使用进化后的专家池激活专家
        
        Args:
            patient: 患者信息
            step1_output: Step-1的输出结果
            top_k: 激活的专家数量
            selection_strategy: 选择策略
                - 'top_fitness': 按适应度选择前K个 (不推荐，可能专科不匹配)
                - 'diversity': 多样性选择
                - 'hybrid': 混合策略（优先匹配专科，同专科选高适应度）
        
        Returns:
            Tuple[base_experts, evolved_prompts]:
                - base_experts: 匹配到的基础专科专家配置
                - evolved_prompts: 对应的进化后Prompt配置
        """
        print("\n" + "-"*60)
        print(f"【Evolved Pool激活】选择策略: {selection_strategy}")
        print("-"*60)
        
        if not self.evolved_pool:
            print("警告: 进化专家池未加载，回退到Step-1路由模式")
            base_experts = self._activate_experts_step1(step1_output)
            return base_experts, [None] * len(base_experts)
        
        # 先用Step-1路由获取推荐的基础专科
        recommended_specialties = self._extract_recommended_specialties(step1_output)
        base_experts = self._match_experts_by_specialties(recommended_specialties)
        
        if not base_experts:
            # 如果没有匹配到，使用全科
            for expert in self.experts:
                if expert['id'] == 'general_practice':
                    base_experts = [expert]
                    break
        
        # 限制专家数量
        base_experts = base_experts[:top_k]
        
        # 根据选择策略分配进化后的Prompt
        evolved_prompts = []
        
        if selection_strategy == "hybrid":
            # 混合策略：严格匹配专科，同专科下选Fitness最高的
            for expert in base_experts:
                # 查找该专科的所有进化专家
                candidates = []
                for evolved in self.evolved_pool:
                    # 匹配逻辑：ID匹配 或 专科名称匹配
                    if (evolved.get('specialty') == expert['specialty'] or 
                        expert['id'] in evolved.get('id', '')):
                        candidates.append(evolved)
                
                if candidates:
                    # 按适应度排序，选最好的
                    best_candidate = max(candidates, key=lambda x: x.get('fitness', 0))
                    evolved_prompts.append(best_candidate)
                else:
                    print(f"  ⚠️ 未找到专科 '{expert['specialty']}' 的进化专家，使用默认Prompt")
                    evolved_prompts.append(None)
                    
        elif selection_strategy == "top_fitness":
            # 按适应度排序，选择前K个 (仅供测试，不保证专科匹配)
            sorted_pool = sorted(self.evolved_pool, key=lambda x: x.get('fitness', 0), reverse=True)
            for i, expert in enumerate(base_experts):
                if i < len(sorted_pool):
                    evolved_prompts.append(sorted_pool[i])
                else:
                    evolved_prompts.append(sorted_pool[0] if sorted_pool else None)
                    
        else: # diversity or others
            # 简单轮询
            for i, expert in enumerate(base_experts):
                idx = i % len(self.evolved_pool)
                evolved_prompts.append(self.evolved_pool[idx])
        
        # 打印激活结果
        print(f"基础专科专家: {[e['name'] for e in base_experts]}")
        print(f"进化Prompt分配:")
        for i, (expert, prompt_config) in enumerate(zip(base_experts, evolved_prompts)):
            if prompt_config:
                fitness = prompt_config.get('fitness', 'N/A')
                prompt_id = prompt_config.get('id', f'prompt_{i}')
                # 检查是否匹配
                match_status = "✅" if prompt_config.get('specialty') == expert['specialty'] else "⚠️(专科不配)"
                print(f"  [{i+1}] {expert['name']} <- {prompt_id} {match_status} (Fitness: {fitness:.4f if isinstance(fitness, float) else fitness})")
            else:
                print(f"  [{i+1}] {expert['name']} <- 默认Prompt")
        
        return base_experts, evolved_prompts
    
    def run_pipeline(self, patient: PatientInfo, top_k: int = 8, 
                     evolved_selection_strategy: str = "hybrid") -> Dict[str, Any]:
        """
        运行完整的四步流水线
        
        Args:
            patient: 患者信息
            top_k: 激活的专家数量
            evolved_selection_strategy: 进化专家池选择策略 ('top_fitness', 'diversity', 'hybrid')
        """
        results = {
            "patient_id": patient.patient_id,
            "activation_mode": self.activation_mode,
            "steps": {}
        }
        
        # Step-1: 会诊/转诊规划
        step1_result = self.step1_route(patient)
        results["steps"]["step1"] = step1_result
        
        # 根据激活模式选择专家
        episode = None
        evolved_prompts = None  # 进化后的Prompt配置
        
        if self.activation_mode == "eep_semantic" and self.expert_pool:
            # EEP混合激活模式：结合语义激活 + Step-1路由
            selected_experts, episode = self._activate_experts_eep(
                patient, 
                top_k=top_k,
                step1_output=step1_result["output"],  # 传入Step-1结果
                step1_weight=0.3  # Step-1推荐占30%权重
            )
            results["steps"]["routing"] = {
                "mode": "eep_hybrid",
                "step1_weight": 0.3,
                "episode_id": episode.episode_id,
                "keywords": list(episode.keywords),
                "system_domain": episode.tags.get('system_domain', []),
                "selected_experts": [e['name'] for e in selected_experts]
            }
            
        elif self.activation_mode == "evolved_pool" and self.evolved_pool:
            # 进化专家池模式：使用GA进化后的最优Prompt
            selected_experts, evolved_prompts = self._activate_experts_evolved(
                patient,
                step1_output=step1_result["output"],
                top_k=top_k,
                selection_strategy=evolved_selection_strategy
            )
            results["steps"]["routing"] = {
                "mode": "evolved_pool",
                "selection_strategy": evolved_selection_strategy,
                "selected_experts": [e['name'] for e in selected_experts],
                "evolved_prompts_used": [
                    {
                        "id": p.get('id', 'unknown'),
                        "fitness": p.get('fitness', 0)
                    } if p else None 
                    for p in (evolved_prompts or [])
                ]
            }
        else:
            # 纯Step-1路由模式
            selected_experts = self._activate_experts_step1(step1_result["output"])
            results["steps"]["routing"] = {
                "mode": "step1_route",
                "selected_experts": [e['name'] for e in selected_experts]
            }
        
        print(f"\n激活的专家 ({len(selected_experts)}): {[e['name'] for e in selected_experts]}")
        
        # Step-2 & Step-3: 对激活的专家进行语义重写和鉴别诊断
        expert_opinions = []
        
        for idx, expert in enumerate(selected_experts):
            print("\n" + "-"*60)
            print(f"【Step-2 & Step-3】[{idx+1}/{len(selected_experts)}] {expert['name']} 分析中...")
            print("-"*60)
            
            # Step-2: 专家语义重写
            step2_result = self.step2_semantic_rewrite(patient, expert)
            print(f"\n[{expert['name']}] 语义重写完成")
            
            # Step-3: 专家鉴别诊断
            # 如果是进化专家池模式，使用对应的进化后Prompt
            custom_prompt = None
            evolved_prompt_id = None
            if evolved_prompts and idx < len(evolved_prompts) and evolved_prompts[idx]:
                custom_prompt = evolved_prompts[idx].get('prompt')
                evolved_prompt_id = evolved_prompts[idx].get('id')
                print(f"  📝 使用进化Prompt: {evolved_prompt_id}")
            
            step3_result = self.step3_diagnosis(
                patient, expert, 
                step2_result["output"],
                custom_prompt_template=custom_prompt
            )
            print(f"\n[{expert['name']}] 鉴别诊断:")
            print(step3_result["output"])
            
            opinion_record = {
                "expert_id": expert['id'],
                "expert_name": expert['name'],
                "specialty": expert['specialty'],
                "rewrite": step2_result["output"],
                "diagnosis": step3_result["output"],
                "output": step3_result["output"]
            }
            
            # 如果使用了进化Prompt，记录下来
            if evolved_prompt_id:
                opinion_record["evolved_prompt_id"] = evolved_prompt_id
            
            expert_opinions.append(opinion_record)
        
        results["steps"]["expert_opinions"] = expert_opinions
        
        # Step-4: 多专家整合裁决
        step4_result = self.step4_aggregate(patient, expert_opinions)
        results["steps"]["step4"] = step4_result
        
        # Step-5 (可选): 分歧检测与专家演化
        if self.activation_mode == "eep_semantic" and self.expert_pool and episode:
            evolution_result = self._detect_and_evolve_with_reanalysis(
                patient,
                step4_result["output"], 
                episode, 
                expert_opinions,
                results
            )
            if evolution_result:
                results["steps"]["evolution"] = evolution_result
        
        # 保存专家池（更新使用统计）
        if self.activation_mode == "eep_semantic" and self.expert_pool:
            self.expert_pool._save_pool()
        
        return results
    
    def _detect_and_evolve_with_reanalysis(self, patient: PatientInfo, step4_output: str, 
                                            episode: DiagnosticEpisode, 
                                            expert_opinions: List[Dict],
                                            results: Dict) -> Optional[Dict]:
        """
        检测诊断分歧，生成新专家，并让新专家参与当前诊断
        
        流程：
        1. 检测Step-4中的分歧
        2. 如有显著分歧，生成新专家单元
        3. 让新专家参与Step-2/3分析
        4. 重新进行Step-4整合（包含新专家意见）
        """
        print("\n" + "-"*60)
        print("【Step-5】分歧检测与专家演化")
        print("-"*60)
        
        # 使用LLM检测分歧
        detect_prompt = f"""分析以下多专家整合裁决结果，判断是否存在需要引入新专家视角的情况。

【Step-4整合结果】
{step4_output}

【参与专家】
{', '.join([e['expert_name'] for e in expert_opinions])}

请分析：
1. 是否存在明显的专家意见分歧？
2. 是否存在当前专家池未能覆盖的重要视角？
3. 是否需要引入新的专家单元来弥补信息缺口？

以JSON格式输出：
{{
    "has_significant_divergence": true/false,
    "divergence_point": "分歧点描述（如无则为空）",
    "conflicting_views": ["观点1", "观点2"],
    "missing_perspective": "缺失的视角描述（如无则为空）",
    "suggested_new_expert": "建议新增的专家类型（中文，如无则为空）",
    "evolution_priority": "high/medium/low/none"
}}

仅输出JSON，不要其他解释。
"""
        
        try:
            result = self._call_llm(detect_prompt)
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1]
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0]
            
            analysis = json.loads(result.strip())
            
            print(f"分歧检测结果:")
            print(f"  - 存在显著分歧: {analysis.get('has_significant_divergence', False)}")
            print(f"  - 分歧点: {analysis.get('divergence_point', '无')}")
            print(f"  - 演化优先级: {analysis.get('evolution_priority', 'none')}")
            
            # 如果分歧显著且优先级高，触发专家演化
            if (analysis.get('has_significant_divergence') and 
                analysis.get('evolution_priority') in ['high', 'medium']):
                
                print(f"\n检测到显著分歧，尝试生成新专家...")
                print(f"  - 缺失视角: {analysis.get('missing_perspective', '未知')}")
                print(f"  - 建议专家: {analysis.get('suggested_new_expert', '未知')}")
                
                # 调用EEP演化机制
                divergence_info = {
                    'divergence_point': analysis.get('divergence_point', ''),
                    'conflicting_views': analysis.get('conflicting_views', []),
                    'missing_perspective': analysis.get('missing_perspective', ''),
                    'suggested_expert': analysis.get('suggested_new_expert', '')
                }
                
                new_expert = self.expert_pool.evolve_from_divergence(divergence_info, episode)
                
                if new_expert:
                    print(f"\n✅ 成功生成新专家单元: {new_expert.name}")
                    print(f"   专科: {new_expert.specialty}")
                    print(f"   关注领域: {', '.join(new_expert.focus_areas)}")
                    
                    # ========== 让新专家参与当前诊断 ==========
                    print("\n" + "="*60)
                    print(f"【新专家参与诊断】{new_expert.name} 加入会诊")
                    print("="*60)
                    
                    new_expert_dict = self._expert_unit_to_dict(new_expert)
                    
                    # Step-2: 新专家语义重写
                    print(f"\n[{new_expert.name}] 进行语义重写...")
                    step2_new = self.step2_semantic_rewrite(patient, new_expert_dict)
                    print(f"[{new_expert.name}] 语义重写完成")
                    
                    # Step-3: 新专家鉴别诊断
                    print(f"\n[{new_expert.name}] 进行鉴别诊断...")
                    step3_new = self.step3_diagnosis(patient, new_expert_dict, step2_new["output"])
                    print(f"\n[{new_expert.name}] 鉴别诊断:")
                    print(step3_new["output"])
                    
                    # 将新专家意见加入列表
                    new_opinion = {
                        "expert_id": new_expert.id,
                        "expert_name": new_expert.name,
                        "specialty": new_expert.specialty,
                        "rewrite": step2_new["output"],
                        "diagnosis": step3_new["output"],
                        "output": step3_new["output"],
                        "is_evolved": True  # 标记为演化生成的专家
                    }
                    
                    # 更新专家意见列表
                    all_opinions = expert_opinions + [new_opinion]
                    results["steps"]["expert_opinions"] = all_opinions
                    
                    # ========== 重新进行Step-4整合 ==========
                    print("\n" + "="*60)
                    print("【Step-4 重新整合】包含新专家意见的裁决")
                    print("="*60)
                    
                    step4_new = self.step4_aggregate(patient, all_opinions)
                    results["steps"]["step4"] = step4_new
                    results["steps"]["step4"]["includes_evolved_expert"] = True
                    
                    return {
                        "divergence_detected": True,
                        "analysis": analysis,
                        "new_expert_created": True,
                        "new_expert": {
                            "id": new_expert.id,
                            "name": new_expert.name,
                            "specialty": new_expert.specialty,
                            "focus_areas": new_expert.focus_areas
                        },
                        "new_expert_diagnosis": step3_new["output"],
                        "reanalysis_performed": True
                    }
                else:
                    print(f"\n⚠️ 未能生成新专家（可能与现有专家重复）")
                    return {
                        "divergence_detected": True,
                        "analysis": analysis,
                        "new_expert_created": False,
                        "reanalysis_performed": False
                    }
            else:
                print("当前无需演化专家池")
                return {
                    "divergence_detected": False,
                    "analysis": analysis,
                    "reanalysis_performed": False
                }
                
        except Exception as e:
            print(f"分歧检测失败: {e}")
            return None


def parse_patient_from_row(row: pd.Series) -> PatientInfo:
    """从DataFrame行解析患者信息"""
    
    # 解析病史JSON
    history_str = row.get('history_clean', row.get('history', '{}'))
    if pd.isna(history_str):
        history_str = '{}'
    
    try:
        if isinstance(history_str, str):
            history = json.loads(history_str)
        else:
            history = {}
    except (json.JSONDecodeError, TypeError):
        history = {}
    
    # 解析各字段
    chief_complaint = history.get('主诉', '')
    hpi = history.get('现病史', '')
    past_history = history.get('既往史', '')
    personal_history = history.get('个人史', '')
    
    # 体格检查
    physical_exam = row.get('physical_examination_compressed', 
                           row.get('physical_examination', ''))
    if pd.isna(physical_exam):
        physical_exam = ''
    
    # 实验室检查
    labs = row.get('labs_compressed', row.get('labs_lite', row.get('labs', '')))
    if pd.isna(labs):
        labs = ''
    
    # 影像学检查
    imaging = row.get('exam_lite', row.get('exam', ''))
    if pd.isna(imaging):
        imaging = ''
    
    # 性别
    gender = row.get('gender', '')
    if gender == 'M':
        gender = '男'
    elif gender == 'F':
        gender = '女'
    
    # 年龄
    age = row.get('age', 0)
    if pd.isna(age):
        age = 0
    
    return PatientInfo(
        patient_id=str(row.get('patient_id', '')),
        gender=gender,
        age=int(age),
        department=str(row.get('normalized_name', row.get('department', ''))),
        chief_complaint=chief_complaint,
        history_of_present_illness=hpi,
        past_history=past_history,
        personal_history=personal_history,
        physical_examination=physical_exam,
        labs=labs,
        imaging=imaging,
        main_diagnosis=str(row.get('main_diagnosis', '')),
        main_diagnosis_icd=str(row.get('main_diagnosis_icd', ''))
    )


class GeneticPromptOptimizer:
    """
    遗传算法优化器：进化诊断Agent的系统提示词(System Prompt)
    
    核心概念映射:
    - Individual (个体): 一个具体的 Prompt 模版
    - Population (种群): 一组 Prompt 模版
    - Fitness (适应度): 在验证集上的诊断准确率 + 召回率
    - Crossover (交叉): 融合两个 Prompt 的特征
    - Mutation (变异): 随机调整 Prompt 的侧重点
    """
    
    def __init__(self, base_prompt: str, pipeline: DiagnosticPipeline, 
                 population_size: int = 32, elitism_count: int = 6):  # 优化：减小默认种群大小
        self.base_prompt = base_prompt
        self.pipeline = pipeline
        self.client = pipeline.client  # 复用OpenAI客户端
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.population = []  # List[Dict]: [{'id': str, 'prompt': str, 'fitness': float, 'stats': Dict}]
        self.generation = 0
        self.best_individual = None
        
        # 新增：全局名人堂 (Hall of Fame)，用于记录历史最优个体
        self.hall_of_fame = []
        
        # 新增：缓存机制
        self.step2_cache = {}  # {(patient_id, expert_id): rewrite_result}
        self.expert_match_cache = {}  # {patient_id: matched_expert}
        
        # 新增：早停机制
        self.fitness_history = []
        self.stagnation_count = 0
        self.max_stagnation = 3  # 连续3代无改进则早停
        
    def _get_matched_expert(self, patient: PatientInfo) -> Dict:
        """智能匹配专家：基于患者科室和症状"""
        if patient.patient_id in self.expert_match_cache:
            return self.expert_match_cache[patient.patient_id]
            
        # 1. 优先按科室匹配
        department = patient.department.lower()
        for expert in self.pipeline.experts:
            if any(dept in department for dept in [expert['specialty'].lower(), expert['name'].lower()]):
                self.expert_match_cache[patient.patient_id] = expert
                return expert
        
        # 2. 如果没匹配到，用全科专家
        for expert in self.pipeline.experts:
            if expert['id'] == 'general_practice':
                self.expert_match_cache[patient.patient_id] = expert
                return expert
                
        # 3. Fallback：返回第一个专家
        self.expert_match_cache[patient.patient_id] = self.pipeline.experts[0]
        return self.pipeline.experts[0]
        
    def initialize_population(self):
        """初始化种群：基于基础Prompt生成多样化的变体"""
        print(f"正在初始化种群 (大小: {self.population_size})...")
        
        # 个体 0 是原始 Prompt
        self.population.append({
            'id': 'gen0_original',
            'prompt': self.base_prompt,
            'fitness': 0.0,
            'stats': {}
        })
        
        # 剩余个体通过变异生成
        for i in range(1, self.population_size):
            try:
                print(f"  - 生成初始个体 {i}/{self.population_size}...")
                mutated_prompt = self._mutate_prompt(self.base_prompt, intensity="high")
                self.population.append({
                    'id': f'gen0_ind{i}',
                    'prompt': mutated_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
            except Exception as e:
                print(f"    警告：个体{i}生成失败: {e}，使用原始Prompt")
                self.population.append({
                    'id': f'gen0_ind{i}_fallback',
                    'prompt': self.base_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
                
    def evaluate_fitness(self, validation_patients: List[PatientInfo], 
                         sample_size: int = 5) -> None:  # 优化：减小默认样本数
        """评估种群适应度（优化版：分科分层采样 + 缓存优化）"""
        print(f"\n开始评估第 {self.generation} 代适应度 (样本数: {sample_size})...")
        
        # === 优化：分层平衡采样 (Stratified Sampling) ===
        # 目的：避免随机采样的病例集中在某一个科室，导致评估偏差
        # 逻辑：按科室对患者分组，从不同科室轮询抽取病例
        
        dept_groups = {}
        for p in validation_patients:
            dept = p.department
            if dept not in dept_groups:
                dept_groups[dept] = []
            dept_groups[dept].append(p)
            
        eval_batch = []
        depts = list(dept_groups.keys())
        import random
        random.shuffle(depts) # 随机打乱科室顺序
        
        # 轮询抽取，直到凑够 sample_size
        while len(eval_batch) < sample_size and len(depts) > 0:
            for dept in depts[:]: # 遍历科室副本
                if len(dept_groups[dept]) > 0:
                    # 从该科室随机取一个
                    p = random.choice(dept_groups[dept])
                    if p not in eval_batch:
                        eval_batch.append(p)
                        # 确保不重复取同一个
                        dept_groups[dept].remove(p) 
                else:
                    depts.remove(dept)
                
                if len(eval_batch) >= sample_size:
                    break
        
        print(f"  - 验证集构成: {', '.join([p.department for p in eval_batch])}")
        # ============================================
        
        # 预计算所有患者的step2结果（缓存优化）
        print("  - 预计算Step2语义重写结果（缓存优化）...")
        for patient in eval_batch:
            expert = self._get_matched_expert(patient)
            cache_key = (patient.patient_id, expert['id'])
            
            if cache_key not in self.step2_cache:
                try:
                    step2_res = self.pipeline.step2_semantic_rewrite(patient, expert)
                    self.step2_cache[cache_key] = step2_res['output']
                except Exception as e:
                    print(f"    警告：Step2失败 {patient.patient_id}: {e}")
                    self.step2_cache[cache_key] = f"患者信息重写失败，原始信息：{patient.to_prompt_string()[:500]}"
        
        # 评估每个个体
        for idx, individual in enumerate(self.population):
            # 如果已经评估过（例如精英保留下来的），跳过
            if individual['fitness'] > 0:
                continue
                
            print(f"  - 评估个体 {idx+1}/{self.population_size} (ID: {individual['id']})...")
            
            total_score = 0
            correct_count = 0
            valid_evaluations = 0
            
            for patient in eval_batch:
                try:
                    expert = self._get_matched_expert(patient)
                    cache_key = (patient.patient_id, expert['id'])
                    
                    # 使用缓存的step2结果
                    step2_output = self.step2_cache.get(cache_key, "")
                    
                    # 运行待评估的 Step 3 (使用个体的 Prompt)
                    step3_res = self.pipeline.step3_diagnosis(
                        patient, expert, 
                        step2_output, 
                        custom_prompt_template=individual['prompt'],
                        auto_retrieve=False  # 关闭自动检索以加速评估
                    )
                    
                    # 计算单次诊断得分 (LLM 裁判)
                    score, is_correct = self._judge_diagnosis(patient, step3_res['output'])
                    total_score += score
                    if is_correct:
                        correct_count += 1
                    valid_evaluations += 1
                        
                except Exception as e:
                    print(f"    警告：患者{patient.patient_id}评估失败: {e}")
                    continue
            
            if valid_evaluations == 0:
                print(f"    个体{individual['id']}评估完全失败，设置最低分")
                individual['fitness'] = 0.0
                individual['stats'] = {'accuracy': 0.0, 'avg_score': 0.0, 'valid_count': 0}
                continue
                
            # 计算平均分
            avg_score = total_score / valid_evaluations
            accuracy = correct_count / valid_evaluations
            
            # 适应度公式: 准确率优先，分数辅助，考虑有效评估数
            fitness = accuracy * 0.7 + (avg_score / 100) * 0.3
            # 如果有效评估数过少，降低适应度
            if valid_evaluations < len(eval_batch) * 0.5:
                fitness *= 0.8  # 惩罚因子
            
            individual['fitness'] = fitness
            individual['stats'] = {
                'accuracy': accuracy, 
                'avg_score': avg_score, 
                'valid_count': valid_evaluations
            }
            print(f"    -> Fitness: {fitness:.4f} (Acc: {accuracy:.0%}, Valid: {valid_evaluations}/{len(eval_batch)})")
            
    def _judge_diagnosis(self, patient: PatientInfo, diagnosis_output: str, 
                        max_retries: int = 2) -> Tuple[float, bool]:
        """使用 LLM 作为裁判评估诊断准确性（增强错误处理）"""
        judge_prompt = f"""
你是一名资深医学专家裁判。请评估AI医生的诊断准确性。

【金标准/真实诊断】
{patient.main_diagnosis} (ICD: {patient.main_diagnosis_icd})

【AI医生诊断输出】
{diagnosis_output}

请评分 (0-100) 并判断是否正确 (True/False)。
评分标准：
- 100: 完全一致，核心诊断正确。
- 80-99: 核心诊断正确，但漏掉次要细节。
- 60-79: 诊断方向正确，但具体病种有偏差。
- 0-59: 误诊或漏诊。

输出格式 JSON: {{"score": 85, "is_correct": true, "reason": "..."}}
仅输出 JSON。
"""
        
        for attempt in range(max_retries):
            try:
                res = self.pipeline._call_llm(judge_prompt)
                # 清理 markdown
                if "```" in res:
                    res = res.split("```json")[-1].split("```")[0] if "```json" in res else res.split("```")[-1].split("```")[0]
                
                data = json.loads(res.strip())
                score = float(data.get('score', 0))
                is_correct = bool(data.get('is_correct', False))
                
                # 基本合理性检查
                if 0 <= score <= 100:
                    return score, is_correct
                else:
                    print(f"    警告：裁判分数异常 {score}，重试...")
                    continue
                    
            except Exception as e:
                print(f"    警告：裁判调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # 所有重试失败，返回保守分数
        print("    裁判评估失败，返回保守分数")
        return 30.0, False

    def _mutate_prompt(self, prompt_text: str, intensity: str = "medium", 
                      max_retries: int = 2) -> str:
        """变异算子：使用 LLM 修改 Prompt（增强错误处理）"""
        mutation_prompt = f"""
你是一个Prompt优化专家。请对以下用于医疗诊断的 System Prompt 进行【变异】操作。
变异强度: {intensity}

【原始 Prompt】
{prompt_text}

【要求】
1. 保持所有格式化占位符不变 (如 {{resource}}, {{patient}}, {{expert}} 等)。
2. 随机改变诊断策略（例如：更激进、更保守、更关注病史、更关注影像、更严谨推理等）。
3. 稍微调整措辞以激发大模型不同的潜能。
4. 直接输出修改后的 Prompt 内容，不要任何解释或 Markdown 标记。
"""
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(mutation_prompt).strip()
                # 基本合理性检查：确保包含必要的占位符
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
                else:
                    print(f"    警告：变异结果缺少占位符，重试...")
                    continue
            except Exception as e:
                print(f"    警告：变异操作失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # 变异失败，返回原始Prompt
        print("    变异失败，返回原始Prompt")
        return prompt_text

    def _crossover_prompts(self, prompt_a: str, prompt_b: str, 
                          max_retries: int = 2) -> str:
        """交叉算子：融合两个 Prompt（增强错误处理）"""
        crossover_prompt = f"""
你是一个Prompt优化专家。请将以下两个医疗诊断 Prompt 进行【交叉/融合】，生成一个新的优良 Prompt。

【父代 Prompt A】
{prompt_a}

【父代 Prompt B】
{prompt_b}

【要求】
1. 提取两者最好的指令部分进行组合。
2. 必须保留所有格式化占位符 ({{resource}}, {{patient}} 等)。
3. 生成逻辑通顺、指令清晰的新 Prompt。
4. 直接输出新 Prompt 内容，不要任何解释。
"""
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(crossover_prompt).strip()
                # 基本合理性检查
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
                else:
                    print(f"    警告：交叉结果缺少占位符，重试...")
                    continue
            except Exception as e:
                print(f"    警告：交叉操作失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
        
        # 交叉失败，随机返回一个父代
        import random
        return random.choice([prompt_a, prompt_b])

    def selection(self):
        """选择算子：精英保留 + 轮盘赌（增加早停检测）"""
        # 排序
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        current_best = sorted_pop[0]['fitness']
        
        # 早停检测
        if len(self.fitness_history) > 0:
            if current_best <= max(self.fitness_history):
                self.stagnation_count += 1
                print(f"  - 无改进 ({self.stagnation_count}/{self.max_stagnation})")
            else:
                self.stagnation_count = 0
        
        self.fitness_history.append(current_best)
        self.best_individual = sorted_pop[0]
        print(f"  - 当前最佳个体: {self.best_individual['id']}, Fitness: {current_best:.4f}")
        
        new_pop = []
        
        # 1. 精英策略 (Elitism)
        for i in range(min(self.elitism_count, len(sorted_pop))):
            new_pop.append(sorted_pop[i])
            print(f"    保留精英: {sorted_pop[i]['id']} (Fitness: {sorted_pop[i]['fitness']:.4f})")
        
        # 2. 补齐剩余
        parents_pool = sorted_pop[:max(len(sorted_pop)//2, 2)]  # 至少保留2个父代
        
        while len(new_pop) < self.population_size:
            import random
            
            # 80% 概率交叉，20% 概率直接变异
            if random.random() < 0.8 and len(parents_pool) >= 2:
                parent_a = random.choice(parents_pool)
                parent_b = random.choice(parents_pool)
                if parent_a == parent_b and len(parents_pool) > 1:
                    continue
                
                child_prompt = self._crossover_prompts(parent_a['prompt'], parent_b['prompt'])
                
                # 孩子也有概率变异
                if random.random() < 0.1:
                    child_prompt = self._mutate_prompt(child_prompt, intensity="low")
                
                new_pop.append({
                    'id': f'gen{self.generation+1}_child_{len(new_pop)}',
                    'prompt': child_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
            else:
                # 变异
                parent = random.choice(parents_pool)
                child_prompt = self._mutate_prompt(parent['prompt'], intensity="medium")
                new_pop.append({
                    'id': f'gen{self.generation+1}_mutant_{len(new_pop)}',
                    'prompt': child_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
        
        self.population = new_pop

    def should_early_stop(self) -> bool:
        """检查是否应该早停"""
        return self.stagnation_count >= self.max_stagnation

    def run_evolution(self, validation_patients: List[PatientInfo], generations: int = 10, top_k_return: int = 32):
        """运行进化主循环（优化版：增加早停和进度保存）"""
        self.initialize_population()
        
        for g in range(generations):
            self.generation = g
            print(f"\n" + "="*60)
            print(f"【进化代数: {g+1}/{generations}】")
            print(f"="*60)
            
            # 评估
            self.evaluate_fitness(validation_patients)
            
            # === 更新名人堂 (Hall of Fame) ===
            for ind in self.population:
                if ind.get('fitness', 0) > 0:
                    # 检查是否已存在（基于Prompt内容去重）
                    exists = False
                    for existing in self.hall_of_fame:
                        if existing['prompt'] == ind['prompt']:
                            exists = True
                            # 如果当前分数更高，更新分数
                            if ind['fitness'] > existing['fitness']:
                                existing['fitness'] = ind['fitness']
                                existing['stats'] = ind['stats']
                                existing['id'] = ind['id']
                            break
                    if not exists:
                        self.hall_of_fame.append(ind.copy())
            
            # 名人堂排序
            self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
            print(f"  📊 名人堂当前收录: {len(self.hall_of_fame)} 个有效专家")
            # ===============================
            
            # 记录最佳
            best = max(self.population, key=lambda x: x['fitness'])
            print(f"\n>>> 第 {g+1} 代最佳 Prompt (Fitness: {best['fitness']:.4f}):")
            print(best['prompt'][:200] + "...")
            
            # 保存每一代的最佳结果
            checkpoint = {
                'generation': g,
                'best_individual': best,
                'fitness_history': self.fitness_history,
                'population_stats': {
                    'avg_fitness': sum(ind['fitness'] for ind in self.population) / len(self.population),
                    'max_fitness': max(ind['fitness'] for ind in self.population),
                    'min_fitness': min(ind['fitness'] for ind in self.population)
                }
            }
            
            with open(f"ga_gen_{g}_checkpoint.json", "w", encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
            # 早停检测
            if g < generations - 1:
                self.selection()
                if self.should_early_stop():
                    print(f"\n🔄 早停触发：连续{self.stagnation_count}代无改进")
                    break
                
        print("\n🏆 进化完成！正在导出最优专家池...")
        
        # 从名人堂中取 Top K (确保都是评估过的高分个体)
        final_pool = self.hall_of_fame[:top_k_return]
        
        # 如果名人堂数量不足 (极少情况)，用当前种群补齐
        if len(final_pool) < top_k_return:
            print(f"⚠️ 名人堂数量 ({len(final_pool)}) 不足 {top_k_return}，使用未评估个体补齐")
            seen_prompts = {p['prompt'] for p in final_pool}
            for ind in self.population:
                if len(final_pool) >= top_k_return:
                    break
                if ind['prompt'] not in seen_prompts:
                    final_pool.append(ind)
                    seen_prompts.add(ind['prompt'])
        
        return final_pool


def main(
    activation_mode: str = "eep_semantic", 
    top_k: int = 8, 
    patient_index: int = 0,
    enable_rag: bool = True,
    enable_experience: bool = True,
    enable_case: bool = True,
    generations: int = 10,
    evolved_pool_path: str = "outputs/moa_optimized_expert_pool_64.json",
    evolved_selection_strategy: str = "hybrid"
):
    """
    主函数
    
    Args:
        activation_mode: 专家激活模式
            - 'eep_semantic': 语义激活模式
            - 'step1_route': 路由激活模式
            - 'evolved_pool': 使用GA进化后的专家池
            - 'train_ga': 遗传算法训练模式
        evolved_pool_path: 进化专家池文件路径
        evolved_selection_strategy: 进化专家选择策略 ('top_fitness', 'diversity', 'hybrid')
    """
    print("="*70)
    print("混合多学科专家智能体诊断系统")
    print(f"模式: {activation_mode}")
    if activation_mode == "evolved_pool":
        print(f"进化专家池: {evolved_pool_path}")
        print(f"选择策略: {evolved_selection_strategy}")
    print("="*70)
    
    print("\n正在加载患者数据...")
    df = pd.read_excel('guilin_inpatient_extracted_10000.xlsx')
    print(f"共加载 {len(df)} 条患者记录")
    
    # 初始化诊断流水线
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic" if activation_mode == "train_ga" else activation_mode,
        evolved_pool_path=evolved_pool_path,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    valid_mask = df['is_history_cleaned'] == True
    valid_df = df[valid_mask]
    
    if len(valid_df) == 0:
        print("未找到有效患者记录，使用第一条记录")
        valid_df = df.iloc[:10] # Fallback
    
    # ==========================
    # 模式 A: 遗传算法训练 (GA)
    # ==========================
    if activation_mode == "train_ga":
        print("\n" + "="*70)
        print("🚀 启动遗传算法进化模式 (Prompt Evolution)")
        print(f"目标: 进化 Step-3 诊断 Prompt | 代数: {generations}")
        print("="*70)
        
        # 准备验证集 (取前50个病人作为训练/验证集)
        train_patients = []
        print("正在解析训练数据...")
        for i in range(min(50, len(valid_df))):
            try:
                p = parse_patient_from_row(valid_df.iloc[i])
                train_patients.append(p)
            except Exception as e:
                pass
        print(f"有效训练样本数: {len(train_patients)}")
        
        # 初始化优化器
        optimizer = GeneticPromptOptimizer(
            base_prompt=system_step3_prompt,
            pipeline=pipeline,
            population_size=32,   # 优化：从128减少到32
            elitism_count=6       # 优化：从13减少到6 (约20%)
        )
        
        # 运行进化
        best_expert_pool = optimizer.run_evolution(train_patients, generations=generations, top_k_return=32)
        
        # 保存最优专家池
        pool_file = "outputs/moa_optimized_expert_pool_64.json"
        with open(pool_file, 'w', encoding='utf-8') as f:
            json.dump(best_expert_pool, f, ensure_ascii=False, indent=2)
            
        print(f"\n🏆 进化完成！最优专家池已保存至 {pool_file}")
        print(f"包含 {len(best_expert_pool)} 个优选专家Prompt")
        
        return best_expert_pool

    # ==========================
    # 模式 B: 单次诊断推理 (Inference)
    # ==========================
    # 支持选择不同患者
    idx = min(patient_index, len(valid_df) - 1)
    sample_row = valid_df.iloc[idx]
    print(f"选择患者索引: {idx}")
    
    patient = parse_patient_from_row(sample_row)
    
    print("\n" + "="*70)
    print("患者基本信息")
    print("="*70)
    print(patient.to_prompt_string())
    
    print("\n" + "="*70)
    print("开始多学科诊断流水线")
    print("="*70)
    
    results = pipeline.run_pipeline(
        patient, 
        top_k=top_k,
        evolved_selection_strategy=evolved_selection_strategy
    )
    
    output_file = f"diagnosis_result_{patient.patient_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print(f"诊断完成！结果已保存至: {output_file}")
    print("="*70)
    
    print("\n【参考】实际临床诊断:")
    print(f"主诊断: {patient.main_diagnosis}")
    print(f"ICD编码: {patient.main_diagnosis_icd}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="混合多学科专家智能体诊断系统")
    parser.add_argument("--mode", type=str, default="eep_semantic",
                       choices=["eep_semantic", "step1_route", "evolved_pool", "train_ga"],
                       help="运行模式: eep_semantic(语义激活), step1_route(路由激活), evolved_pool(进化专家池), train_ga(遗传进化训练)")
    parser.add_argument("--top_k", type=int, default=8,
                       help="激活的专家数量 (推荐 5-15)")
    parser.add_argument("--patient", type=int, default=0,
                       help="患者索引 (仅推理模式)")
    parser.add_argument("--generations", type=int, default=10,
                       help="进化代数 (仅GA模式)")
    
    # 进化专家池参数
    parser.add_argument("--evolved_pool", type=str, default="outputs/moa_optimized_expert_pool_64.json",
                       help="进化专家池文件路径 (仅evolved_pool模式)")
    parser.add_argument("--evolved_strategy", type=str, default="hybrid",
                       choices=["top_fitness", "diversity", "hybrid"],
                       help="进化专家选择策略: top_fitness(按适应度), diversity(多样性), hybrid(混合)")
    
    # 知识检索参数
    parser.add_argument("--enable_rag", type=bool, default=True,
                       help="启用RAG指南检索")
    parser.add_argument("--enable_experience", type=bool, default=True,
                       help="启用经验库检索")
    parser.add_argument("--enable_case", type=bool, default=True,
                       help="启用病例库检索")
    parser.add_argument("--no_rag", action="store_true",
                       help="禁用RAG指南检索")
    parser.add_argument("--no_experience", action="store_true",
                       help="禁用经验库检索")
    parser.add_argument("--no_case", action="store_true",
                       help="禁用病例库检索")
    
    args = parser.parse_args()
    
    # 处理禁用标志
    enable_rag = not args.no_rag
    enable_experience = not args.no_experience
    enable_case = not args.no_case
    
    main(
        activation_mode=args.mode, 
        top_k=args.top_k, 
        patient_index=args.patient,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
        generations=args.generations,
        evolved_pool_path=args.evolved_pool,
        evolved_selection_strategy=args.evolved_strategy
    )

