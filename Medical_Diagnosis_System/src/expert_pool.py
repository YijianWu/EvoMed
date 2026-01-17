"""
可扩展、可演化的专家资源池 (Evolving Expert Pool, EEP)

支持：
1. 专家单元的形式化表示 (ExpertUnit)
2. Episode条件化的专家激活 (Episode-Conditioned Activation)
3. 专家资源池的增量构建与演化 (Incremental Evolution)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import hashlib


# API配置
API_BASE_URL = "https://yunwu.ai/v1"
API_KEY = "sk-CCoYJEJcm2mL4YH7uRRw9DPgXQj2f8873F1D98uXtuwclUwW"
EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class ExpertMetadata:
    """专家结构化元信息 M_k"""
    specialty_tags: List[str]           # 学科标签
    applicable_stages: List[str]        # 适用诊断阶段: ["初诊", "复诊", "住院期", "急诊"]
    risk_preference: str                # 风险取向: "保守", "中立", "激进"
    target_populations: List[str]       # 典型适用人群: ["成人", "老年", "儿童", "孕产妇"]
    system_focus: List[str]             # 关注系统: ["消化系统", "心血管系统", ...]
    confidence_domains: List[str]       # 高置信领域
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0                # 使用次数统计
    effectiveness_score: float = 0.5    # 有效性评分 [0, 1]


@dataclass
class ExpertUnit:
    """
    专家单元 E_k = (D_k, M_k, V_k)
    
    D_k: 专家背景与视角描述（自然语言）
    M_k: 结构化元信息
    V_k: 向量表示（用于语义匹配）
    """
    id: str                             # 唯一标识
    name: str                           # 专家名称
    specialty: str                      # 主专科
    description: str                    # D_k: 专家背景与视角描述
    focus_areas: List[str]              # 关注重点
    thinking_patterns: List[str]        # 典型思维路径
    metadata: ExpertMetadata            # M_k: 结构化元信息
    embedding: Optional[List[float]] = None  # V_k: 向量表示
    
    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertUnit':
        """从字典创建（用于反序列化）"""
        metadata = ExpertMetadata(**data.pop('metadata'))
        return cls(metadata=metadata, **data)
    
    def get_description_for_prompt(self) -> str:
        """生成用于prompt的专家描述"""
        return f"""
专家角色: {self.name}
专业领域: {self.specialty}
专业描述: {self.description}
关注重点: {', '.join(self.focus_areas)}
思维路径: {', '.join(self.thinking_patterns)}
风险偏好: {self.metadata.risk_preference}
擅长人群: {', '.join(self.metadata.target_populations)}
"""


@dataclass
class DiagnosticEpisode:
    """
    诊断Episode ε = (S_ε, K_ε, T_ε)
    
    S_ε: 病例最小充分证据摘要
    K_ε: 关键词集合
    T_ε: 标签集合（症状域、系统域、风险域等）
    """
    episode_id: str
    summary: str                        # S_ε: 证据摘要
    keywords: Set[str]                  # K_ε: 关键词集合
    tags: Dict[str, List[str]]          # T_ε: 标签集合
    stage: str = "初诊"                 # 诊断阶段
    embedding: Optional[List[float]] = None  # 向量表示
    
    @classmethod
    def from_patient_info(cls, patient_info: Any, episode_id: str = None) -> 'DiagnosticEpisode':
        """从患者信息创建Episode"""
        if episode_id is None:
            episode_id = hashlib.md5(patient_info.patient_id.encode()).hexdigest()[:12]
        
        # 构建证据摘要
        summary = f"""
患者：{patient_info.gender}，{patient_info.age}岁
主诉：{patient_info.chief_complaint}
现病史：{patient_info.history_of_present_illness[:500] if patient_info.history_of_present_illness else '无'}
既往史：{patient_info.past_history[:300] if patient_info.past_history else '无'}
"""
        
        # 提取关键词（简化版，实际可用NLP提取）
        keywords = set()
        text = f"{patient_info.chief_complaint} {patient_info.history_of_present_illness}"
        # 常见症状关键词
        symptom_keywords = ["疼痛", "腹痛", "腹胀", "头痛", "胸闷", "发热", "咳嗽", 
                          "恶心", "呕吐", "腹泻", "便秘", "出血", "水肿", "乏力"]
        for kw in symptom_keywords:
            if kw in text:
                keywords.add(kw)
        
        # 构建标签
        tags = {
            "symptom_domain": [],    # 症状域
            "system_domain": [],     # 系统域
            "risk_domain": [],       # 风险域
            "population": []         # 人群特征
        }
        
        # 系统域推断
        system_keywords = {
            "消化系统": ["腹痛", "腹胀", "恶心", "呕吐", "腹泻", "便秘", "胃", "肠"],
            "心血管系统": ["胸闷", "心悸", "心脏", "血压", "心力衰竭", "心房颤动"],
            "呼吸系统": ["咳嗽", "咳痰", "气促", "呼吸困难", "肺"],
            "泌尿系统": ["尿频", "尿急", "尿痛", "肾", "膀胱"],
            "神经系统": ["头痛", "头晕", "意识", "肢体麻木", "脑"],
            "内分泌系统": ["糖尿病", "甲状腺", "血糖"],
            "生殖系统": ["月经", "妊娠", "子宫", "卵巢"],
        }
        
        for system, kws in system_keywords.items():
            for kw in kws:
                if kw in text:
                    if system not in tags["system_domain"]:
                        tags["system_domain"].append(system)
                    break
        
        # 人群特征
        if patient_info.age < 18:
            tags["population"].append("儿童")
        elif patient_info.age >= 65:
            tags["population"].append("老年")
        else:
            tags["population"].append("成人")
            
        if patient_info.gender == "女":
            tags["population"].append("女性")
        else:
            tags["population"].append("男性")
        
        return cls(
            episode_id=episode_id,
            summary=summary,
            keywords=keywords,
            tags=tags,
            stage="初诊"
        )


class EvolvingExpertPool:
    """
    可演化专家资源池 (Evolving Expert Pool, EEP)
    
    实现：
    1. 专家单元管理（CRUD）
    2. Episode条件化专家激活
    3. 增量构建与演化
    """
    
    def __init__(self, pool_path: str = "expert_pool.json"):
        self.pool_path = Path(pool_path)
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        self.experts: Dict[str, ExpertUnit] = {}
        self._load_pool()
    
    def _load_pool(self):
        """加载专家资源池"""
        if self.pool_path.exists():
            with open(self.pool_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for expert_data in data.get('experts', []):
                    expert = ExpertUnit.from_dict(expert_data)
                    self.experts[expert.id] = expert
            print(f"已加载 {len(self.experts)} 个专家单元")
        else:
            self._initialize_default_experts()
            self._save_pool()
    
    def _save_pool(self):
        """保存专家资源池"""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'experts': [e.to_dict() for e in self.experts.values()]
        }
        with open(self.pool_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding生成失败: {e}")
            return [0.0] * 1536  # 返回零向量作为fallback
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        v1, v2 = np.array(v1), np.array(v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _initialize_default_experts(self):
        """初始化默认专家单元"""
        default_experts = [
            ExpertUnit(
                id="obgyn",
                name="妇产科专家",
                specialty="妇产科",
                description="擅长女性生殖系统疾病的诊治及围产期保健，关注女性全生命周期健康",
                focus_areas=["孕产期管理", "妇科肿瘤", "月经失调", "生殖道炎症"],
                thinking_patterns=["生殖内分泌轴评估", "妊娠相关症状鉴别", "盆腔占位分析"],
                metadata=ExpertMetadata(
                    specialty_tags=["妇产科", "妇科", "产科"],
                    applicable_stages=["初诊", "复诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["女性", "孕产妇"],
                    system_focus=["生殖系统"],
                    confidence_domains=["妇科肿瘤", "异常子宫出血", "妊娠并发症"]
                )
            ),
            ExpertUnit(
                id="gastroenterology",
                name="消化内科专家",
                specialty="消化内科",
                description="擅长食管、胃肠、肝胆胰等消化系统疾病的内科诊治及内镜检查",
                focus_areas=["胃炎与溃疡", "肝脏疾病", "功能性胃肠病", "消化道出血"],
                thinking_patterns=["消化道症状定位", "肝功能异常分析", "内镜指征评估"],
                metadata=ExpertMetadata(
                    specialty_tags=["消化内科", "消化科", "胃肠内科"],
                    applicable_stages=["初诊", "复诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["消化系统"],
                    confidence_domains=["消化性溃疡", "肝硬化", "炎症性肠病"]
                )
            ),
            ExpertUnit(
                id="pediatrics",
                name="儿科专家",
                specialty="儿科",
                description="专注新生儿至青少年时期的生长发育及疾病诊治，关注儿童特有的生理病理特点",
                focus_areas=["呼吸道感染", "生长发育评估", "小儿消化系统", "新生儿护理"],
                thinking_patterns=["年龄分层评估", "发育里程碑对照", "儿童用药剂量换算"],
                metadata=ExpertMetadata(
                    specialty_tags=["儿科", "小儿科", "儿内科"],
                    applicable_stages=["初诊", "复诊", "住院期", "急诊"],
                    risk_preference="保守",
                    target_populations=["儿童", "新生儿", "青少年"],
                    system_focus=["呼吸系统", "消化系统", "神经系统"],
                    confidence_domains=["小儿呼吸道感染", "发育迟缓", "儿童腹泻"]
                )
            ),
            ExpertUnit(
                id="endocrinology",
                name="内分泌科专家",
                specialty="内分泌科",
                description="擅长激素分泌异常及代谢性疾病的诊断与长期管理",
                focus_areas=["糖尿病管理", "甲状腺疾病", "骨质疏松", "肥胖与代谢综合征"],
                thinking_patterns=["激素轴功能评估", "代谢指标综合分析", "长期并发症预防"],
                metadata=ExpertMetadata(
                    specialty_tags=["内分泌科", "内分泌", "代谢科"],
                    applicable_stages=["初诊", "复诊"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["内分泌系统"],
                    confidence_domains=["糖尿病", "甲状腺功能异常", "骨代谢疾病"]
                )
            ),
            ExpertUnit(
                id="hepatobiliary_surgery",
                name="肝胆外科专家",
                specialty="肝胆外科",
                description="擅长肝脏、胆道及胰腺疾病的外科手术治疗及围手术期管理",
                focus_areas=["胆石症", "肝脏肿瘤", "胰腺炎", "胆道梗阻"],
                thinking_patterns=["肝胆影像解读", "手术适应症评估", "围手术期风险分层"],
                metadata=ExpertMetadata(
                    specialty_tags=["肝胆外科", "肝胆胰外科", "肝外科"],
                    applicable_stages=["复诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["消化系统"],
                    confidence_domains=["胆囊结石", "肝癌", "胰腺肿瘤"]
                )
            ),
            ExpertUnit(
                id="orthopedics",
                name="骨科专家",
                specialty="骨科",
                description="擅长骨骼、关节、肌肉及韧带等运动系统疾病的诊断、复位及手术治疗",
                focus_areas=["骨折创伤", "关节炎", "脊柱疾病", "运动损伤"],
                thinking_patterns=["骨折分型评估", "关节功能评分", "手术vs保守治疗决策"],
                metadata=ExpertMetadata(
                    specialty_tags=["骨科", "骨外科", "创伤骨科"],
                    applicable_stages=["初诊", "急诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年", "儿童"],
                    system_focus=["运动系统"],
                    confidence_domains=["四肢骨折", "退行性关节病", "腰椎间盘突出"]
                )
            ),
            ExpertUnit(
                id="respiratory",
                name="呼吸内科专家",
                specialty="呼吸内科",
                description="擅长呼吸系统感染、气道疾病及肺部肿瘤的内科诊断与治疗",
                focus_areas=["慢性阻塞性肺病", "哮喘管理", "肺部结节", "肺部感染"],
                thinking_patterns=["肺功能评估", "影像学征象解读", "感染vs肿瘤鉴别"],
                metadata=ExpertMetadata(
                    specialty_tags=["呼吸内科", "呼吸科", "肺科"],
                    applicable_stages=["初诊", "复诊", "住院期", "急诊"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["呼吸系统"],
                    confidence_domains=["肺炎", "COPD", "肺癌筛查"]
                )
            ),
            ExpertUnit(
                id="emergency",
                name="急诊科专家",
                specialty="急诊科",
                description="擅长急性病、创伤及各类危重症的初步评估、急救复苏与分诊",
                focus_areas=["生命体征维持", "急性中毒", "多发伤", "心肺复苏"],
                thinking_patterns=["ABCDE评估", "危重症识别", "快速分诊决策"],
                metadata=ExpertMetadata(
                    specialty_tags=["急诊科", "急诊", "急诊医学科"],
                    applicable_stages=["急诊"],
                    risk_preference="激进",
                    target_populations=["成人", "老年", "儿童"],
                    system_focus=["心血管系统", "呼吸系统", "神经系统"],
                    confidence_domains=["急性胸痛", "休克", "急性腹痛"]
                )
            ),
            ExpertUnit(
                id="urology",
                name="泌尿外科专家",
                specialty="泌尿外科",
                description="擅长泌尿系统及男性生殖系统疾病的外科及微创治疗",
                focus_areas=["泌尿系结石", "前列腺疾病", "泌尿系肿瘤", "尿路感染"],
                thinking_patterns=["尿路梗阻评估", "PSA解读", "微创vs开放手术选择"],
                metadata=ExpertMetadata(
                    specialty_tags=["泌尿外科", "泌尿科"],
                    applicable_stages=["初诊", "复诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年", "男性"],
                    system_focus=["泌尿系统", "生殖系统"],
                    confidence_domains=["肾结石", "前列腺增生", "膀胱癌"]
                )
            ),
            ExpertUnit(
                id="nephrology",
                name="肾内科专家",
                specialty="肾内科",
                description="擅长肾脏疾病的内科诊治，包括肾功能不全、肾炎及透析管理",
                focus_areas=["慢性肾病管理", "肾炎综合征", "透析治疗", "电解质紊乱"],
                thinking_patterns=["肾功能分期评估", "蛋白尿分析", "肾脏替代治疗决策"],
                metadata=ExpertMetadata(
                    specialty_tags=["肾内科", "肾科"],
                    applicable_stages=["初诊", "复诊", "住院期"],
                    risk_preference="保守",
                    target_populations=["成人", "老年"],
                    system_focus=["泌尿系统"],
                    confidence_domains=["慢性肾病", "糖尿病肾病", "IgA肾病"]
                )
            ),
            ExpertUnit(
                id="general_practice",
                name="全科医学专家",
                specialty="全科医学科",
                description="提供综合性、连续性的基本医疗服务，擅长未分化疾病的初诊与常见病管理",
                focus_areas=["健康查体解读", "常见病初诊", "双向转诊", "慢病长期随访"],
                thinking_patterns=["全人评估", "问题列表优先级排序", "转诊时机判断"],
                metadata=ExpertMetadata(
                    specialty_tags=["全科医学科", "全科", "全科医学", "综合内科"],
                    applicable_stages=["初诊", "复诊"],
                    risk_preference="保守",
                    target_populations=["成人", "老年", "儿童"],
                    system_focus=["消化系统", "心血管系统", "呼吸系统", "内分泌系统"],
                    confidence_domains=["常见病初诊", "慢病管理", "健康咨询"]
                )
            ),
            ExpertUnit(
                id="gastro_surgery",
                name="胃肠外科专家",
                specialty="胃肠外科",
                description="擅长胃、小肠、结直肠及肛门疾病的手术治疗，特别是肿瘤及急腹症处理",
                focus_areas=["胃肠道肿瘤", "阑尾炎", "肠梗阻", "疝气修补"],
                thinking_patterns=["急腹症定位", "肿瘤TNM分期", "手术切除范围评估"],
                metadata=ExpertMetadata(
                    specialty_tags=["胃肠外科", "普外科", "结直肠外科"],
                    applicable_stages=["急诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["消化系统"],
                    confidence_domains=["结直肠癌", "急性阑尾炎", "肠梗阻"]
                )
            ),
            ExpertUnit(
                id="cardiothoracic_surgery",
                name="胸心外科专家",
                specialty="胸心血管外科",
                description="擅长胸腔内器官的复杂外科手术治疗",
                focus_areas=["肺癌手术", "心脏瓣膜病", "冠脉搭桥", "主动脉夹层"],
                thinking_patterns=["心功能术前评估", "肺癌分期与手术方案", "主动脉病变分型"],
                metadata=ExpertMetadata(
                    specialty_tags=["胸心血管外科", "胸外科", "心外科", "心胸外科"],
                    applicable_stages=["住院期", "急诊"],
                    risk_preference="激进",
                    target_populations=["成人", "老年"],
                    system_focus=["心血管系统", "呼吸系统"],
                    confidence_domains=["肺癌根治术", "瓣膜置换", "CABG"]
                )
            ),
            ExpertUnit(
                id="oncology",
                name="肿瘤科专家",
                specialty="肿瘤科",
                description="擅长各类良恶性肿瘤的内科综合治疗，包括化疗、靶向治疗及免疫治疗",
                focus_areas=["放化疗方案", "肿瘤筛查", "癌痛管理", "多学科会诊(MDT)"],
                thinking_patterns=["肿瘤分期评估", "治疗方案选择", "预后评估"],
                metadata=ExpertMetadata(
                    specialty_tags=["肿瘤科", "肿瘤内科", "肿瘤外科", "放疗科"],
                    applicable_stages=["复诊", "住院期"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["消化系统", "呼吸系统", "生殖系统"],
                    confidence_domains=["实体瘤化疗", "靶向治疗", "免疫检查点抑制剂"]
                )
            ),
            ExpertUnit(
                id="cardiology",
                name="心血管内科专家",
                specialty="心血管内科",
                description="擅长心脏及血管疾病的内科诊疗与介入治疗，关注心血管风险防控",
                focus_areas=["高血压", "冠心病", "心律失常", "心力衰竭"],
                thinking_patterns=["心血管风险分层", "心电图解读", "介入vs药物治疗决策"],
                metadata=ExpertMetadata(
                    specialty_tags=["心血管内科", "心内科", "心脏科", "心脏内科"],
                    applicable_stages=["初诊", "复诊", "住院期", "急诊"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["心血管系统"],
                    confidence_domains=["急性冠脉综合征", "心衰管理", "房颤抗凝"]
                )
            ),
            ExpertUnit(
                id="neurology",
                name="神经内科专家",
                specialty="神经内科",
                description="擅长神经系统疾病的诊断与内科治疗",
                focus_areas=["脑血管病", "头痛", "癫痫", "帕金森病"],
                thinking_patterns=["神经系统定位诊断", "卒中时间窗评估", "认知功能评估"],
                metadata=ExpertMetadata(
                    specialty_tags=["神经内科", "神经科"],
                    applicable_stages=["初诊", "复诊", "住院期", "急诊"],
                    risk_preference="中立",
                    target_populations=["成人", "老年"],
                    system_focus=["神经系统"],
                    confidence_domains=["缺血性卒中", "偏头痛", "帕金森病"]
                )
            ),
            ExpertUnit(
                id="nutrition",
                name="营养科专家",
                specialty="营养科",
                description="擅长临床营养评估、围手术期营养支持及慢性病饮食管理",
                focus_areas=["营养不良评估", "肠内肠外营养", "术前营养优化", "代谢支持"],
                thinking_patterns=["营养风险筛查", "能量需求计算", "营养干预方案制定"],
                metadata=ExpertMetadata(
                    specialty_tags=["营养科", "临床营养科"],
                    applicable_stages=["住院期", "复诊"],
                    risk_preference="保守",
                    target_populations=["成人", "老年", "儿童"],
                    system_focus=["消化系统", "内分泌系统"],
                    confidence_domains=["营养不良", "围手术期营养", "肠内营养"]
                )
            ),
        ]
        
        # 初始化每个专家的向量表示
        print("正在初始化专家向量表示...")
        for expert in default_experts:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description} {' '.join(expert.focus_areas)}"
            expert.embedding = self._get_embedding(embedding_text)
            self.experts[expert.id] = expert
        
        print(f"已初始化 {len(self.experts)} 个默认专家单元")
    
    def activate_experts(self, episode: DiagnosticEpisode, 
                        top_k: int = 5,
                        semantic_weight: float = 0.4,
                        tag_weight: float = 0.4,
                        stage_weight: float = 0.2) -> List[ExpertUnit]:
        """
        Episode条件化专家激活
        
        A(ε) = {E_k ∈ E | relevance(E_k, ε) > θ}
        
        综合考虑：
        1. 语义相似性: sim(V_k, V_ε)
        2. 标签匹配: |T_ε ∩ M_k.tags|
        3. 诊断阶段一致性
        """
        # 获取Episode的向量表示
        if episode.embedding is None:
            episode.embedding = self._get_embedding(episode.summary)
        
        scores = []
        
        for expert_id, expert in self.experts.items():
            # 1. 语义相似性
            if expert.embedding:
                semantic_sim = self._cosine_similarity(episode.embedding, expert.embedding)
            else:
                semantic_sim = 0.0
            
            # 2. 标签匹配
            tag_score = 0.0
            total_tags = 0
            matched_tags = 0
            
            # 系统域匹配
            for system in episode.tags.get("system_domain", []):
                total_tags += 1
                if system in expert.metadata.system_focus:
                    matched_tags += 1
            
            # 人群匹配
            for pop in episode.tags.get("population", []):
                total_tags += 1
                if pop in expert.metadata.target_populations:
                    matched_tags += 1
            
            if total_tags > 0:
                tag_score = matched_tags / total_tags
            
            # 3. 诊断阶段一致性
            stage_score = 1.0 if episode.stage in expert.metadata.applicable_stages else 0.3
            
            # 综合得分
            total_score = (
                semantic_weight * semantic_sim +
                tag_weight * tag_score +
                stage_weight * stage_score
            )
            
            scores.append((expert, total_score, {
                'semantic': semantic_sim,
                'tag': tag_score,
                'stage': stage_score
            }))
        
        # 按得分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k专家
        activated = []
        print(f"\n专家激活得分排序 (top-{top_k}):")
        for i, (expert, score, details) in enumerate(scores[:top_k]):
            print(f"  {i+1}. {expert.name}: {score:.3f} "
                  f"(语义:{details['semantic']:.3f}, 标签:{details['tag']:.3f}, 阶段:{details['stage']:.3f})")
            activated.append(expert)
            # 更新使用统计
            expert.metadata.usage_count += 1
            expert.metadata.updated_at = datetime.now().isoformat()
        
        return activated
    
    def activate_experts_hybrid(
        self, 
        episode: DiagnosticEpisode,
        step1_recommended: List[str],
        step1_weight: float = 0.3,
        top_k: int = 5,
        semantic_weight: float = 0.4,
        tag_weight: float = 0.4,
        stage_weight: float = 0.2
    ) -> List[ExpertUnit]:
        """
        混合激活模式：结合EEP语义激活 + Step-1路由推荐
        
        总得分 = (1 - step1_weight) * EEP得分 + step1_weight * Step1匹配得分
        
        Args:
            episode: 诊断Episode
            step1_recommended: Step-1推荐的专科列表
            step1_weight: Step-1推荐的权重 (0-1)
            top_k: 返回专家数量
        """
        # 获取Episode的向量表示
        if episode.embedding is None:
            episode.embedding = self._get_embedding(episode.summary)
        
        # 构建专科别名映射（用于匹配Step-1推荐）
        specialty_aliases = {
            "心内科": ["心内科", "心血管内科", "心脏内科", "循环内科"],
            "消化内科": ["消化内科", "消化科", "胃肠内科"],
            "肿瘤科": ["肿瘤科", "肿瘤内科", "肿瘤外科", "肿瘤学"],
            "呼吸内科": ["呼吸内科", "呼吸科", "肺科"],
            "妇产科": ["妇产科", "妇科", "产科"],
            "急诊科": ["急诊科", "急诊"],
            "神经内科": ["神经内科", "神经科"],
            "肝胆外科": ["肝胆外科", "肝胆胰外科"],
            "泌尿外科": ["泌尿外科", "泌尿科"],
            "骨科": ["骨科", "骨外科"],
            "胃肠外科": ["胃肠外科", "普外科", "胃肠道外科"],
            "内分泌科": ["内分泌科", "内分泌", "代谢科"],
            "肾内科": ["肾内科", "肾脏内科", "肾科"],
            "全科": ["全科", "全科医学", "全科医学科"],
        }
        
        # 将Step-1推荐标准化
        normalized_step1 = set()
        for spec in step1_recommended:
            normalized_step1.add(spec)
            for key, aliases in specialty_aliases.items():
                if spec in aliases or key in spec:
                    normalized_step1.update(aliases)
                    normalized_step1.add(key)
        
        scores = []
        
        for expert_id, expert in self.experts.items():
            # ========== EEP得分 ==========
            # 1. 语义相似性
            if expert.embedding:
                semantic_sim = self._cosine_similarity(episode.embedding, expert.embedding)
            else:
                semantic_sim = 0.0
            
            # 2. 标签匹配
            tag_score = 0.0
            total_tags = 0
            matched_tags = 0
            
            for system in episode.tags.get("system_domain", []):
                total_tags += 1
                if system in expert.metadata.system_focus:
                    matched_tags += 1
            
            for pop in episode.tags.get("population", []):
                total_tags += 1
                if pop in expert.metadata.target_populations:
                    matched_tags += 1
            
            if total_tags > 0:
                tag_score = matched_tags / total_tags
            
            # 3. 诊断阶段一致性
            stage_score = 1.0 if episode.stage in expert.metadata.applicable_stages else 0.3
            
            # EEP综合得分
            eep_score = (
                semantic_weight * semantic_sim +
                tag_weight * tag_score +
                stage_weight * stage_score
            )
            
            # ========== Step-1匹配得分 ==========
            step1_score = 0.0
            if step1_recommended:
                # 检查专家专科是否在Step-1推荐中
                expert_specialty = expert.specialty
                expert_tags = expert.metadata.specialty_tags
                
                # 直接匹配
                if expert_specialty in normalized_step1:
                    step1_score = 1.0
                else:
                    # 模糊匹配
                    for tag in expert_tags:
                        if tag in normalized_step1:
                            step1_score = 1.0
                            break
                    
                    # 部分匹配
                    if step1_score == 0:
                        for rec in step1_recommended:
                            if rec in expert_specialty or expert_specialty in rec:
                                step1_score = 0.8
                                break
                            for tag in expert_tags:
                                if rec in tag or tag in rec:
                                    step1_score = 0.6
                                    break
            
            # ========== 混合得分 ==========
            if step1_recommended:
                total_score = (1 - step1_weight) * eep_score + step1_weight * step1_score
            else:
                total_score = eep_score  # 无Step-1推荐时退化为纯EEP
            
            scores.append((expert, total_score, {
                'eep': eep_score,
                'step1': step1_score,
                'semantic': semantic_sim,
                'tag': tag_score,
                'stage': stage_score
            }))
        
        # 按得分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k专家
        activated = []
        print(f"\n混合激活得分排序 (top-{top_k}, Step-1权重={step1_weight:.0%}):")
        for i, (expert, score, details) in enumerate(scores[:top_k]):
            step1_mark = "✓" if details['step1'] > 0 else " "
            print(f"  {i+1}. [{step1_mark}] {expert.name}: {score:.3f} "
                  f"(EEP:{details['eep']:.3f}, Step1:{details['step1']:.1f})")
            activated.append(expert)
            expert.metadata.usage_count += 1
            expert.metadata.updated_at = datetime.now().isoformat()
        
        return activated
    
    def add_expert(self, expert: ExpertUnit, save: bool = True) -> bool:
        """
        增量添加新专家单元
        
        E' = E ∪ {E_{new}}
        """
        # 首先检查ID是否已存在
        if expert.id in self.experts:
            print(f"跳过: 专家ID '{expert.id}' 已存在")
            return False
        
        # 检查是否存在语义重复
        if expert.embedding is None:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description}"
            expert.embedding = self._get_embedding(embedding_text)
        
        # 找到最相似的现有专家
        max_sim = 0.0
        most_similar_expert = None
        
        for existing in self.experts.values():
            if existing.embedding:
                sim = self._cosine_similarity(expert.embedding, existing.embedding)
                if sim > max_sim:
                    max_sim = sim
                    most_similar_expert = existing
        
        # 如果相似度过高，拒绝添加（只打印一次警告）
        if max_sim > 0.95 and most_similar_expert:
            print(f"跳过: 新专家 '{expert.name}' 与现有专家 '{most_similar_expert.name}' 高度相似 (相似度={max_sim:.3f})")
            return False
        
        self.experts[expert.id] = expert
        print(f"已添加新专家单元: {expert.name} (id={expert.id})")
        
        if save:
            self._save_pool()
        
        return True
    
    def update_expert(self, expert_id: str, updates: Dict[str, Any], save: bool = True) -> bool:
        """
        更新专家单元（局部维度细化）
        """
        if expert_id not in self.experts:
            print(f"专家 {expert_id} 不存在")
            return False
        
        expert = self.experts[expert_id]
        
        for key, value in updates.items():
            if hasattr(expert, key):
                setattr(expert, key, value)
            elif hasattr(expert.metadata, key):
                setattr(expert.metadata, key, value)
        
        expert.metadata.updated_at = datetime.now().isoformat()
        
        # 如果描述更新了，重新计算embedding
        if 'description' in updates or 'focus_areas' in updates:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description}"
            expert.embedding = self._get_embedding(embedding_text)
        
        if save:
            self._save_pool()
        
        return True
    
    def evolve_from_divergence(self, 
                               divergence_info: Dict[str, Any],
                               episode: DiagnosticEpisode) -> Optional[ExpertUnit]:
        """
        基于诊断分歧生成新专家单元
        
        当多专家间出现稳定且可复现的推理分歧时，
        系统可生成新的专家单元以覆盖该分歧视角
        """
        divergence_point = divergence_info.get('divergence_point', '')
        conflicting_views = divergence_info.get('conflicting_views', [])
        missing_perspective = divergence_info.get('missing_perspective', '')
        suggested_expert = divergence_info.get('suggested_expert', '')
        
        if not divergence_point:
            return None
        
        # 生成新专家描述（强制中文，严格按照建议生成）
        new_expert_prompt = f"""你是一个医学专家系统设计师。请基于诊断分歧情况，设计一个新的专家来弥补当前专家池的不足。

【重要】所有输出必须使用中文！

【分歧点】
{divergence_point}

【缺失的视角】
{missing_perspective}

【建议新增的专家类型】
{suggested_expert}

【冲突观点】
{json.dumps(conflicting_views, ensure_ascii=False, indent=2)}

【病例背景】
{episode.summary}

请严格按照"建议新增的专家类型"生成对应的专家配置。
输出JSON格式，所有字段必须使用中文（id除外）：

{{
    "id": "英文ID，如 tumor_specialist 或 gastroenterology_functional",
    "name": "中文专家名称，如 肿瘤科专家、功能性胃肠病专家",
    "specialty": "中文专科名称，如 肿瘤科、功能性胃肠病科",
    "description": "中文描述，50-100字，说明该专家的专业特长",
    "focus_areas": ["中文关注领域1", "中文关注领域2", "中文关注领域3"],
    "thinking_patterns": ["中文思维路径1", "中文思维路径2"],
    "risk_preference": "保守/中立/激进"
}}

注意：
1. name和specialty必须与"建议新增的专家类型"对应，例如建议"肿瘤学专家"则生成"肿瘤科专家"
2. 所有描述、领域、思维路径必须使用中文
3. 仅输出JSON，不要任何解释
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个医学专家系统设计师，必须使用中文输出所有专家配置（id字段除外）。"},
                    {"role": "user", "content": new_expert_prompt}
                ],
                temperature=0.3  # 降低温度，使输出更稳定
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]
            
            config = json.loads(result.strip())
            
            # 验证是否为中文名称
            if not any('\u4e00' <= c <= '\u9fff' for c in config.get('name', '')):
                # 如果name不是中文，使用建议的专家类型
                if suggested_expert and any('\u4e00' <= c <= '\u9fff' for c in suggested_expert):
                    config['name'] = suggested_expert if '专家' in suggested_expert else f"{suggested_expert}专家"
                    config['specialty'] = suggested_expert.replace('专家', '').replace('学', '')
            
            # 创建新专家单元
            new_expert = ExpertUnit(
                id=config['id'],
                name=config['name'],
                specialty=config['specialty'],
                description=config['description'],
                focus_areas=config['focus_areas'],
                thinking_patterns=config.get('thinking_patterns', []),
                metadata=ExpertMetadata(
                    specialty_tags=[config['specialty']],
                    applicable_stages=["初诊", "复诊", "住院期"],
                    risk_preference=config.get('risk_preference', '中立'),
                    target_populations=["成人", "老年"],
                    system_focus=episode.tags.get('system_domain', []),
                    confidence_domains=config['focus_areas'][:3]
                )
            )
            
            # 尝试添加
            if self.add_expert(new_expert):
                return new_expert
            
        except Exception as e:
            print(f"生成新专家失败: {e}")
        
        return None
    
    def get_expert_by_specialty(self, specialty: str) -> Optional[ExpertUnit]:
        """根据专科名称获取专家"""
        for expert in self.experts.values():
            if specialty in expert.metadata.specialty_tags or specialty == expert.specialty:
                return expert
        return None
    
    def get_all_experts(self) -> List[ExpertUnit]:
        """获取所有专家"""
        return list(self.experts.values())
    
    def get_specialty_mapping(self) -> Dict[str, ExpertUnit]:
        """获取专科名称到专家的映射"""
        mapping = {}
        for expert in self.experts.values():
            mapping[expert.specialty] = expert
            for tag in expert.metadata.specialty_tags:
                mapping[tag] = expert
        return mapping


# 便捷函数
def create_default_pool(pool_path: str = "expert_pool.json") -> EvolvingExpertPool:
    """创建默认专家资源池"""
    return EvolvingExpertPool(pool_path)


if __name__ == "__main__":
    # 测试专家资源池
    print("="*60)
    print("可演化专家资源池 (EEP) 测试")
    print("="*60)
    
    pool = EvolvingExpertPool()
    
    print(f"\n专家资源池包含 {len(pool.experts)} 个专家单元")
    
    # 模拟一个诊断Episode
    from dataclasses import dataclass as dc
    
    @dc
    class MockPatient:
        patient_id: str = "test001"
        gender: str = "女"
        age: int = 45
        chief_complaint: str = "腹痛、腹胀3天"
        history_of_present_illness: str = "3天前无明显诱因出现腹痛、腹胀，伴恶心，无呕吐、发热"
        past_history: str = "高血压病史5年"
    
    patient = MockPatient()
    episode = DiagnosticEpisode.from_patient_info(patient)
    
    print(f"\n诊断Episode:")
    print(f"  关键词: {episode.keywords}")
    print(f"  系统域: {episode.tags['system_domain']}")
    print(f"  人群: {episode.tags['population']}")
    
    # 激活相关专家
    print("\n激活专家...")
    activated = pool.activate_experts(episode, top_k=5)
    
    print(f"\n激活的专家: {[e.name for e in activated]}")

