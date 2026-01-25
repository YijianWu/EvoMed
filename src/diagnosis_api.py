"""
诊断服务API接口

输入：
- 患者病历文本（整合后的）
- 肠鸣音预测结果（来自其他模型）
- ECG预测结果（来自其他模型）

处理流程：
- Step 1: 科室路由 → 从28专家池中选择对应专家
- Step 2: 专家语义重写
- Step 3: 专家诊断（使用28专家池中的evolved_prompt）
- Step 4: 多专家整合裁决（含危险分层，分两部分输出）

输出：
- Part 1: 患者可见的诊断结果（诊断Top5、诊断依据、鉴别诊断、检查建议）
- Part 2: 仅医生可见的危险分层（危险分层、居家观测、就诊意见、是否需要急救）
"""

import json
import re
import os
import argparse
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

# =============================================================================
# 内置 Prompts (System Prompts)
# =============================================================================

system_step1_prompt = """\
你是一个"多学科会诊与就医路径规划"辅助智能体，
帮助用户把症状信息结构化，并给出"下一步应该优先咨询哪些专科/专家"的建议

【医疗资源】
{resource}

【患者信息】
{patient}

【可用专科列表】
- 妇产科（孕产期管理、妇科肿瘤、月经失调、生殖道炎症）
- 消化内科（胃炎与溃疡、肝脏疾病、功能性胃肠病、消化道出血）
- 儿科（呼吸道感染、生长发育评估、小儿消化系统、新生儿护理）
- 内分泌科（糖尿病管理、甲状腺疾病、骨质疏松、肥胖与代谢综合征）
- 肝胆外科（胆石症、肝脏肿瘤、胰腺炎、胆道梗阻）
- 骨科（骨折创伤、关节炎、脊柱疾病、运动损伤）
- 呼吸内科（慢性阻塞性肺病、哮喘管理、肺部结节、肺部感染）
- 急诊科（生命体征维持、急性中毒、多发伤、心肺复苏）
- 泌尿外科（泌尿系结石、前列腺疾病、泌尿系肿瘤、尿路感染）
- 全科医学科（健康查体解读、常见病初诊、双向转诊、慢病长期随访）
- 胃肠外科（胃肠道肿瘤、阑尾炎、肠梗阻、疝气修补）
- 胸心血管外科（肺癌手术、心脏瓣膜病、冠脉搭桥、主动脉夹层）
- 肿瘤科（放化疗方案、肿瘤筛查、癌痛管理、多学科会诊）
- 心血管内科（高血压、冠心病、心律失常、心力衰竭）

注：每个专科都有2位专家（变体1和变体2），系统会自动分配最合适的专家。

【会诊/转诊推理原则（必须遵守）】
- 采用"问题列表 + 风险分层 + 可能系统/器官来源 + 专科映射"的框架
- 按：症状-体征-时间轴-诱因/缓解因素-既往史-用药/过敏-家族史-暴露史-人群特征（年龄/妊娠等）综合评估
- 避免"单一症状→单一疾病"的简单映射；优先识别高危但可逆、需尽快排除的问题
- 给出"学科优先级"，从上述可用专科列表中选择最相关的专科
- 对证据强度做隐性分级（常见/较常见/少见/需优先排除）
- 可以推荐多个专科进行多学科会诊，但应明确优先级

【输出结构（必须严格遵循）】
建议优先咨询的专家/科室（按临床优先级排序）
- 每一项：推荐专科（必须从上述可用专科列表中选择） + 推荐理由（对应哪些症状/风险因素/时间轴）+ 该专科主要要排除/确认的大类问题（不下最终诊断）
- 明确哪些属于"需要优先排除的高风险情况"，并对应到"急诊/当日就医/近期门诊"
"""

system_step2_prompt = """\
你是一个专家智能体，
你的任务不是进行医疗诊断或给出治疗建议，
而是结合专家的临床经验特点，对患者信息进行医学语义层面的重写、抽象和结构化表达，
用于后续的病历经验检索与相似病例匹配。

【医疗资源】
{resource}

【原始患者信息】
{patient}

【专家信息（用于风格与关注点建模）】
{expert}

【重写与抽象原则】
- 将患者的自然语言描述，改写为更符合该专家临床经验与思维习惯的医学表述
- 提取可用于病例检索的关键医学要素
- 强化与专家常关注的疾病谱、风险因素、症状组合相关的信息

【输出结构（必须严格遵循）】
一、患者关键信息的医学化重写  
- 用更偏临床记录/病历语言，重写患者主诉与病程  
- 体现专家常用的表达方式或关注重点  

二、结构化检索要素摘要  
- 主要症状（规范医学表述）  
- 症状时间特征（起病方式、持续时间、变化趋势）  
- 伴随症状或阴性线索（如有）  
- 已知危险因素 / 背景信息  
- 明确不确定或信息缺失之处  
"""

system_step3_prompt = """\
你是一个医疗诊断辅助智能体，
用于在非替代医疗行为的前提下，辅助用户理解其健康相关症状，
提供循证医学背景下的可能诊断、解释、风险评估及下一步行动建议。

【医疗资源】
{resource}

【患者信息】
{patient}

【专家信息】
{expert}

【参考资料】
{reference}

【推理与分析原则】
- 使用鉴别诊断框架进行推理
- 按照症状-体征-时间轴-危险因素进行因果与概率评估
- 避免单一症状直接映射到单一疾病
- 优先考虑高危但可逆、需排除的疾病
- 对证据强度进行隐性分级（如：常见 / 较常见 / 少见 / 需排除）

【输出结构（必须严格遵循）】
一、可能的诊断和解释（鉴别诊断，按可能性或临床优先级排序）  
- 每一项需简要说明支持理由
- 明确哪些情况属于需要优先排除的高风险疾病

二、潜在风险与警示信号
- 明确指出哪些症状组合提示风险升高
- 区分一般风险与紧急风险
- 说明忽视风险可能带来的后果（概括性）

三、下一步检查建议  
- 可能需要的检查类型（如影像学、实验室检查，非具体指标）
"""

system_step4_prompt_v2 = """\
你是一个医疗诊断智能体，
你的职责是对多位医学专家基于循证医学给出的诊断意见进行系统性整合与裁决，
形成一个一致、可解释、风险可控的综合诊断结论。

【医疗资源】
{resource}

【患者信息】
{patient}

【专家信息与专家意见】
{experts}

【汇总与裁决原则】
- 以"鉴别诊断共识"为核心，综合多专家意见而非简单投票
- 优先保留多名专家一致支持的诊断结论
- 对高危、需紧急排除的疾病采取"宁可过度提醒、不遗漏"的策略
- 当专家意见冲突时：
  - 比较各自的证据充分性与适用性
  - 明确指出分歧点及其原因（证据不足 / 角度不同 / 信息缺失）
- 对诊断不确定性进行保守表达，避免过度确定化结论

【输出结构（必须严格遵循）】

一、危险分层
# 提示词: 急诊预检分诊分级标准

你将根据以下"急诊预检分诊分级标准"对患者病情进行标记。标记应根据患者的主诊断、病史、体格检查、检验信息和检查信息进行评估，并综合考虑以下五个级别的描述细则：

**急诊预检分诊分级标准：**

1. **急危（I级）**
   - **级别描述**：正在或即将发生的生命威胁或病情恶化，需要立即进行积极干预。
   - **客观评估指标**：
     - 心率 > 180 次/分钟 或 < 40 次/分钟。
     - 收缩压 < 70 mmHg 或急性血压降低，较平素血压低 30~60 mmHg。
     - SpO2 < 80% 且呼吸急促，且经吸氧无法改善。
     - 腋温 > 41℃。
     - POCT指标如血糖 < 3.33 mmol/L，血钾 > 7.0 mmol/L。
   - **人工评定指标**：
     - 心搏停止、呼吸停止或节律不稳定，气道无法维持。
     - 休克、明确心肌梗死。
     - 急性意识障碍（GCS < 9）或无反应，只有疼痛刺激反应。
     - 癫痫持续状态、复合伤（需要快速团队应对）、急性药物过量。
     - 严重的精神行为异常（如自伤或他伤行为，需立即药物控制）。
     - 严重休克的儿童/婴儿、小儿惊厥等。
   - **响应程序**：立即进行评估和救治，安排患者进入复苏区。

2. **急重（II级）**
   - **级别描述**：病情危重或迅速恶化，如短时间内不能进行治疗则危及生命或造成严重的器官功能衰竭，或短时间内治疗可对预后产生重大影响。
   - **客观评估指标**：
     - 心率：150-180 次/分钟 或 40~50 次/分钟。
     - 收缩压：> 200 mmHg 或 70~80 mmHg。
     - SpO2：80%~90% 且呼吸急促，且经吸氧无法改善。
     - 发热伴粒细胞减少。
     - POCT指标、ECG提示急性心肌梗死等。
   - **人工评定指标**：
     - 严重呼吸困难，气道无法保护。
     - 循环障碍，皮肤湿冷、花斑，灌注差或怀疑脓毒症。
     - 昏睡（强烈刺激下有防御反应）。
     - 急性脑卒中，类似心脏因素的胸痛。
     - 严重的胸腹疼痛或不明原因的严重疼痛伴大汗。
     - 胸腹痛伴急性心梗、急性肺栓塞、主动脉夹层、消化道穿孔等高风险疾病。
     - 活动性或严重失血、大创伤，如大骨折、截肢。
     - 严重的精神行为异常（暴力或攻击行为，威胁自身或他人，需被约束）。
   - **响应程序**：立即监护生命体征，10分钟内得到救治，安排患者进入抢救区。

3. **急症（III级）**
   - **级别描述**：存在潜在的生命威胁，如短时间内不进行干预，病情可进展至威胁生命或产生十分不利的结局。
   - **客观评估指标**：
     - 心率：100-150 次/分钟 或 50~55 次/分钟。
     - 收缩压：180-200 mmHg 或 80~90 mmHg。
     - SpO2：90%~94% 且呼吸急促，且经吸氧无法改善。
   - **人工评定指标**：
     - 急性哮喘，但血压、脉搏稳定。
     - 嗜睡（可唤醒，无刺激下转入睡眠）。
     - 间断癫痫发作。
     - 中等程度非心源性胸痛，腹痛（特别是 > 65 岁且无高危因素）。
     - 中重度疼痛（4~6分）、中度失血、头外伤、外伤后肢体感觉运动异常等。
     - 持续呕吐或脱水。
     - 精神行为异常：有自残风险、急性精神错乱、思维混乱、焦虑、抑郁、潜在攻击性等。
     - 稳定的新生儿。
   - **响应程序**：优先诊治，安排患者在优先诊疗区候诊，30分钟内接诊；若候诊时间大于30分钟，需再次评估。

4. **亚急症（IV级）**
   - **级别描述**：存在潜在的严重性，患者一定时间内没有给予治疗，可能会恶化或出现不利的结局，或症状会加重或持续时间延长。
   - **客观评估指标**：生命体征平稳。
   - **人工评定指标**：
     - 吸入异物，但无呼吸困难。
     - 吞咽困难，无呼吸困难。
     - 轻度呕吐或腹泻，无脱水。
     - 中等程度疼痛，但没有明显危险特征。
     - 轻度肋骨损伤，或无呼吸困难的胸部损伤。
     - 非特异性轻度腹痛，轻微出血、轻微头部损伤等。
     - 小的肢体创伤，生命体征正常，轻中度疼痛。
     - 关节热胀，轻度肿痛。
     - 精神行为异常，但对自身或他人无直接威胁。
   - **响应程序**：顺序就诊，60分钟内得到接诊；若候诊时间大于60分钟，需再次评估。

5. **非急症（V级）**
   - **级别描述**：慢性或非常轻微的症状，即便等待一段时间再进行治疗也不会对结局产生大的影响。
   - **客观评估指标**：生命体征平稳。
   - **人工评定指标**：
     - 症状轻微，生命体征平稳。
     - 低危病史且目前无症状或症状轻微，如轻微疼痛、擦伤等。
     - 稳定恢复期患者复诊，慢性症状患者。
     - 微小伤口或无需缝合的小擦伤、裂伤。
     - 轻微精神行为异常，无严重危险。
     - 稳定恢复期或无症状患者复诊，仅开药或医疗证明。
   - **响应程序**：顺序就诊，除非病情变化，否则候诊时间较长（2~4小时）；若候诊时间大于4小时，可再次评估。

**任务要求：**
- 综合参考患者的主诊断、病史、体格检查、检验信息和检查信息，分析病情，判断病人属于哪个级别（急危、急重、急症、亚急症、非急症）。
- 如果病历信息符合多个级别的标准，优先选择最高的分级。例如，如果符合急危（I级）和急重（II级）的标准，标记为急危。
- 如果病历信息无法明确归类到某一分级，根据病情的轻重缓急进行合理判断。

- 基于多专家意见和临床风险评估，对患者当前病情进行危险程度分层
- 明确标注：
  - 居家观测：适合居家监测的症状和体征，监测频率和注意事项
  - 就诊意见：建议的就诊时机、科室和就诊方式（门诊/急诊）
  - 是否需要急救：是否存在需要立即急救的紧急情况，明确判断标准

二、诊断Top5
- 列出最重要的5个诊断，按临床优先级和共识强度排序
- 每个诊断用简洁的标题表示

三、诊断依据（不少于100字）
- 为每个Top5诊断提供详细的依据说明
- 包括：
  - 支持该诊断的专家角色
  - 关键临床证据和检查结果，详细列出支持该诊断的临床证据，如症状、体征、影像学检查结果、实验室数据等。解释每一项证据与诊断的关系，以及它们如何帮助确定该诊断。
  - 详细的推理逻辑和临床相关性，提供详细的推理过程，解释如何根据症状、检查结果和历史资料得出该诊断，并分析这些推理与患者临床表现的相关性。
- 明确标注诊断的确信度（高度可能、一般可能、需排除）

四、鉴别诊断
- 对Top5诊断进行系统性鉴别
- 说明每个诊断需排除的相似或鉴别疾病
- 列出鉴别诊断的关键特征和区分要点
- 指出当前证据无法确定的诊断点

五、建议的检查或检验
- 基于诊断需要和专家共识，给出具体的检查建议
- 按优先级排序（紧急、近期、择期）
- 明确每个检查的目的和预期价值
- 说明哪些检查有助于解决当前诊断不确定性
- 仅整合专家已提及的检查方向，不新增未经专家支持的检查
"""

API_BASE_URL = "https://yunwu.ai/v1"
API_KEY = "sk-mZ1tJ8giPu2WqauY5SivguiTVJmFolWNAkBQ4i5Y3Lh2jxVL"
MODEL_NAME = "gpt-4o"

# 28专家池路径 - 动态获取
# 优先查找顺序:
# 1. 环境变量 EXPERT_POOL_PATH
# 2. 当前脚本同级目录下的 optimized_expert_pool_28.json
# 3. 当前脚本同级目录下的 outputs/optimized_expert_pool_28.json
# 4. 当前脚本上级目录下的 outputs/optimized_expert_pool_28.json (原项目结构)

def get_default_expert_pool_path():
    # 1. Env Var
    env_path = os.environ.get("EXPERT_POOL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    candidates = [
        # 同级目录
        os.path.join(current_dir, "optimized_expert_pool_28.json"),
        # 同级 outputs
        os.path.join(current_dir, "outputs", "optimized_expert_pool_28.json"),
        # 上级 outputs (原项目结构)
        os.path.join(current_dir, "../outputs/optimized_expert_pool_28.json"),
        # 当前工作目录
        os.path.join(os.getcwd(), "optimized_expert_pool_28.json")
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
            
    # Default fallback
    return os.path.join(current_dir, "../outputs/optimized_expert_pool_28.json")

EXPERT_POOL_28_PATH = get_default_expert_pool_path()


@dataclass
class BowelSoundResult:
    """肠鸣音预测结果"""
    fold: str = ""
    pid: str = ""
    pred: int = 0  # 0 或 1
    prob_0: float = 0.5
    prob_1: float = 0.5
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BowelSoundResult':
        return cls(
            fold=str(data.get('fold', '')),
            pid=str(data.get('pid', '')),
            pred=int(data.get('pred', 0)),
            prob_0=float(data.get('prob_0', 0.5)),
            prob_1=float(data.get('prob_1', 0.5))
        )
    
    def to_clinical_text(self) -> str:
        """转换为临床描述文本"""
        pred_label = "异常" if self.pred == 1 else "正常"
        confidence = max(self.prob_0, self.prob_1) * 100
        
        text = f"【肠鸣音检测结果】\n"
        text += f"- 预测结果: {pred_label}\n"
        text += f"- 置信度: {confidence:.1f}%\n"
        
        if self.pred == 1:
            text += f"- 异常概率: {self.prob_1*100:.1f}%\n"
            text += f"- 提示: 肠鸣音可能存在异常，建议结合临床症状进一步评估\n"
        else:
            text += f"- 正常概率: {self.prob_0*100:.1f}%\n"
            text += f"- 提示: 肠鸣音未见明显异常\n"
        
        return text


@dataclass
class ECGResult:
    """ECG预测结果"""
    pid: str = ""
    path: str = ""
    pred_id: int = 0
    pred: bool = False  # False=正常, True=异常
    conf: float = 0.5
    topk: List[Tuple[str, float]] = field(default_factory=list)
    probs_json: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ECGResult':
        topk_raw = data.get('topk', [])
        if isinstance(topk_raw, str):
            try:
                topk = eval(topk_raw)
            except:
                topk = []
        else:
            topk = topk_raw if topk_raw else []
        
        pred_val = data.get('pred', False)
        if isinstance(pred_val, bool):
            pred = pred_val
        elif isinstance(pred_val, str):
            pred = pred_val.lower() == 'true'
        else:
            pred = bool(pred_val)
        
        return cls(
            pid=str(data.get('pid', '')),
            path=str(data.get('path', '')),
            pred_id=int(data.get('pred_id', 0)),
            pred=pred,
            conf=float(data.get('conf', 0.5)),
            topk=topk,
            probs_json=str(data.get('probs_json', ''))
        )
    
    def to_clinical_text(self) -> str:
        """转换为临床描述文本"""
        pred_label = "异常" if self.pred else "正常"
        confidence = self.conf * 100
        
        text = f"【心电图(ECG)AI分析结果】\n"
        text += f"- AI预测结果: {pred_label}\n"
        text += f"- 置信度: {confidence:.1f}%\n"
        
        if self.pred:
            text += f"- 提示: 心电图可能存在异常，建议专业医师复阅并结合临床判断\n"
        else:
            text += f"- 提示: 心电图AI分析未见明显异常\n"
        
        if self.topk:
            text += f"- 详细概率:\n"
            for label, prob in self.topk:
                label_cn = "异常" if label == "True" else "正常"
                text += f"    {label_cn}: {prob*100:.1f}%\n"
        
        return text


@dataclass
class DiagnosisOutput:
    """诊断输出结果"""
    patient_visible: Dict[str, Any] = field(default_factory=dict)
    doctor_only: Dict[str, Any] = field(default_factory=dict)
    expert_opinions: List[Dict] = field(default_factory=list)
    raw_output: str = ""
    
    def to_patient_response(self) -> Dict[str, Any]:
        """返回仅患者可见的内容"""
        return {
            "status": "success",
            "data": self.patient_visible,
            "notice": "以上为AI辅助诊断建议，仅供参考，请以专业医生诊断为准。"
        }
    
    def to_doctor_response(self) -> Dict[str, Any]:
        """返回完整内容（包含医生专属部分）"""
        return {
            "status": "success",
            "patient_info": self.patient_visible,
            "risk_assessment": self.doctor_only,
            "expert_opinions": self.expert_opinions,
            "raw_output": self.raw_output
        }

    def to_management_response(self, app_id: str = "", conversation_id: str = "", clinic_code: str = "") -> Dict[str, Any]:
        """返回医生端/管理端展示格式"""
        
        # 构建总结
        risk_level = self.doctor_only.get('risk_level', '')
        top1_diagnosis = "未知"
        if self.patient_visible.get('diagnosis_top5'):
             lines = self.patient_visible['diagnosis_top5'].split('\n')
             for line in lines:
                 if line.strip():
                     top1_diagnosis = line.strip()
                     break
        
        summary = f"风险等级: {risk_level}。初步诊断倾向于: {top1_diagnosis}。建议结合临床进一步检查。"

        # 诊断结果
        diag_result = self.patient_visible.get('diagnosis_top5', '')
        
        # 鉴别诊断及病情分析 (合并 诊断依据 和 鉴别诊断)
        condition_analysis = ""
        if self.patient_visible.get('diagnosis_basis'):
             condition_analysis += f"【诊断依据】\n{self.patient_visible['diagnosis_basis']}\n\n"
        if self.patient_visible.get('differential_diagnosis'):
             condition_analysis += f"【鉴别诊断】\n{self.patient_visible['differential_diagnosis']}"
        
        # 下一步检查建议
        suggestions = self.patient_visible.get('suggested_examinations', '')

        return {
            "type": "DiagnosticMessage",
            "app_id": app_id or str(uuid.uuid4()),
            "summary_label": "总结",
            "summary_value": summary,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "clinic_code": clinic_code, 
            "diagnostic_result_label": "诊断结果",
            "diagnostic_result_value": diag_result,
            "condition_analysis_label": "鉴别诊断及病情分析",
            "condition_analysis_value": condition_analysis.strip(),
            "suggestions_examinations_label": "下一步检查建议",
            "suggestions_examinations_value": suggestions,
            "diffDiagnosis_and_conditionAnalysis_label": "鉴别诊断及病情分析",
            "diffDiagnosis_and_conditionAnalysis_value": condition_analysis.strip()
        }


class DiagnosisAPI:
    """
    诊断服务API
    
    使用训练好的28个最佳专家池进行诊断：
    - Step 1 推荐科室 → 从28专家池中选择对应专家
    - 每个专家使用其优化后的 evolved_prompt
    """
    
    def __init__(self, expert_pool_path: str = EXPERT_POOL_28_PATH):
        """
        初始化诊断API
        
        Args:
            expert_pool_path: 28专家池文件路径
        """
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
        
        self.resource = """【医疗资源】
可用科室：内科、外科、急诊科、心内科、消化内科、呼吸内科、神经内科、肿瘤科、骨科、泌尿外科、肝胆外科、妇产科、儿科等
知识库：医学指南库、临床经验库、相似病例库
说明：本系统仅提供诊断辅助，不替代专业医疗行为。"""
        
        # 加载28专家池
        print("="*60)
        print("初始化诊断服务API - 28专家池")
        print("="*60)
        
        self.expert_pool_28 = self._load_expert_pool(expert_pool_path)
        if self.expert_pool_28:
            print(f"✅ 成功加载28专家池: {len(self.expert_pool_28)} 位专家")
            # 列出专科分布
            specialties = set(e.get('specialty', '') for e in self.expert_pool_28)
            print(f"   覆盖专科: {len(specialties)} 个")
        else:
            raise RuntimeError(f"无法加载28专家池: {expert_pool_path}")
        
        # 构建专科到专家的映射
        self.specialty_to_experts = self._build_specialty_mapping()
    
    def _load_expert_pool(self, path: str) -> Optional[List[Dict]]:
        """加载28专家池"""
        if not os.path.exists(path):
            print(f"  ❌ 专家池文件不存在: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            
            # 确保每个专家都有prompt字段
            for expert in pool:
                if 'evolved_prompt' in expert and 'prompt' not in expert:
                    expert['prompt'] = expert['evolved_prompt']
            
            return pool
        except Exception as e:
            print(f"  ❌ 加载专家池失败: {e}")
            return None
    
    def _build_specialty_mapping(self) -> Dict[str, List[Dict]]:
        """构建专科到专家列表的映射（每个专科2位专家）"""
        mapping = {}
        
        for expert in self.expert_pool_28:
            specialty = expert.get('specialty', '')
            if specialty:
                if specialty not in mapping:
                    mapping[specialty] = []
                mapping[specialty].append(expert)
        
        # 按fitness排序（每个专科内）
        for specialty in mapping:
            mapping[specialty] = sorted(
                mapping[specialty], 
                key=lambda x: x.get('fitness', 0), 
                reverse=True
            )
        
        # 添加专科别名
        aliases = {
            "妇产科": ["妇科", "产科"],
            "消化内科": ["消化科", "胃肠内科"],
            "心血管内科": ["心内科", "心脏科"],
            "呼吸内科": ["呼吸科", "肺科"],
            "肝胆外科": ["肝胆胰外科"],
            "胃肠外科": ["普外科"],
            "泌尿外科": ["泌尿科"],
            "全科医学科": ["全科", "综合内科"],
        }
        
        for main_specialty, alias_list in aliases.items():
            if main_specialty in mapping:
                for alias in alias_list:
                    mapping[alias] = mapping[main_specialty]
        
        return mapping
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """调用LLM API"""
        import time
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个专业的医疗诊断辅助智能体。"},
                        {"role": "user", "content": prompt}
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
    
    def _step1_route(self, patient_info: str) -> Tuple[str, List[str]]:
        """
        Step 1: 科室路由
        返回: (路由结果文本, 推荐专科列表)
        """
        print("\n【Step 1】科室路由（从28专家池中选择）...")
        
        # 列出可用专科供Step1参考
        available_specialties = list(set(e.get('specialty', '') for e in self.expert_pool_28))
        
        prompt = system_step1_prompt.format(
            resource=self.resource + f"\n\n【可选专科】\n{', '.join(available_specialties)}",
            patient=patient_info
        )
        
        result = self._call_llm(prompt)
        
        # 从28专家池的专科中提取推荐
        recommended = self._extract_specialties_from_pool(result)
        print(f"  推荐专科: {recommended}")
        
        return result, recommended
    
    def _extract_specialties_from_pool(self, step1_output: str) -> List[str]:
        """从Step1输出中提取推荐的专科（仅限28专家池中的专科）"""
        # 获取28专家池中的所有专科
        pool_specialties = set(e.get('specialty', '') for e in self.expert_pool_28)
        
        # 添加别名
        all_keywords = list(pool_specialties)
        alias_map = {
            "心内科": "心血管内科",
            "消化科": "消化内科",
            "普外科": "胃肠外科",
            "肝胆胰外科": "肝胆外科",
            "全科": "全科医学科",
            "妇科": "妇产科",
            "产科": "妇产科",
        }
        all_keywords.extend(alias_map.keys())
        
        found = []
        for keyword in all_keywords:
            if keyword in step1_output:
                # 标准化为28专家池中的专科名
                normalized = alias_map.get(keyword, keyword)
                if normalized in pool_specialties and normalized not in found:
                    found.append(normalized)
        
        # 如果没找到，默认使用全科
        if not found:
            found = ["全科医学科"]
        
        return found
    
    def _select_experts(self, recommended_specialties: List[str], max_experts: int = 5) -> List[Dict]:
        """
        从28专家池中选择专家
        每个推荐的专科选择fitness最高的专家
        """
        selected = []
        selected_ids = set()
        
        for specialty in recommended_specialties:
            if len(selected) >= max_experts:
                break
            
            experts = self.specialty_to_experts.get(specialty, [])
            for expert in experts:  # 已按fitness排序
                if expert['id'] not in selected_ids:
                    selected.append(expert)
                    selected_ids.add(expert['id'])
                    print(f"  选中: {expert['name']} (fitness={expert.get('fitness', 0):.3f})")
                    break
        
        # 如果专家太少，补充高fitness专家
        if len(selected) < 2:
            sorted_pool = sorted(self.expert_pool_28, key=lambda x: x.get('fitness', 0), reverse=True)
            for expert in sorted_pool:
                if expert['id'] not in selected_ids:
                    selected.append(expert)
                    selected_ids.add(expert['id'])
                    print(f"  补充: {expert['name']} (fitness={expert.get('fitness', 0):.3f})")
                    if len(selected) >= 3:
                        break
        
        return selected
    
    def _step2_rewrite(self, patient_info: str, expert: Dict) -> str:
        """Step 2: 专家语义重写"""
        expert_desc = f"""
专家角色: {expert.get('name', '专家')}
专业领域: {expert.get('specialty', '')}
专业描述: {expert.get('description', '')}
关注重点: {', '.join(expert.get('focus_areas', []))}
"""
        
        prompt = system_step2_prompt.format(
            resource=self.resource,
            patient=patient_info,
            expert=expert_desc
        )
        
        return self._call_llm(prompt)
    
    def _step3_diagnose(self, patient_info: str, expert: Dict, rewritten_info: str) -> str:
        """
        Step 3: 专家诊断
        使用专家的evolved_prompt（28专家池中的优化prompt）
        """
        expert_desc = f"""
专家角色: {expert.get('name', '专家')}
专业领域: {expert.get('specialty', '')}
专业描述: {expert.get('description', '')}
关注重点: {', '.join(expert.get('focus_areas', []))}

【专家视角下的患者信息重写】
{rewritten_info}
"""
        
        # 使用专家的evolved_prompt
        template = expert.get('prompt') or expert.get('evolved_prompt') or system_step3_prompt
        
        prompt = template.format(
            resource=self.resource,
            patient=patient_info,
            expert=expert_desc,
            reference="【参考资料】暂无"
        )
        
        return self._call_llm(prompt)
    
    def _step4_aggregate(self, patient_info: str, expert_opinions: List[Dict]) -> str:
        """
        Step 4: 多专家整合裁决
        使用新prompt，包含危险分层
        """
        print("\n【Step 4】多专家整合裁决（含危险分层）...")
        
        experts_summary = ""
        for opinion in expert_opinions:
            experts_summary += f"""
{'='*40}
【{opinion['expert_name']}】的诊断意见
专科: {opinion['specialty']}
{'='*40}
{opinion['diagnosis']}

"""
        
        prompt = system_step4_prompt_v2.format(
            resource=self.resource,
            patient=patient_info,
            experts=experts_summary
        )
        
        return self._call_llm(prompt)
    
    def _parse_output(self, raw_output: str) -> Tuple[Dict, Dict]:
        """解析Step4输出，分离患者可见和医生专属部分"""
        sections = {
            '危险分层': '',
            '诊断Top5': '',
            '诊断依据': '',
            '鉴别诊断': '',
            '建议的检查或检验': ''
        }
        
        current_section = None
        lines = raw_output.split('\n')
        current_content = []
        
        for line in lines:
            found_section = False
            for section_name in sections.keys():
                if section_name in line and any(m in line for m in ['一、', '二、', '三、', '四、', '五、']):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = section_name
                    current_content = []
                    found_section = True
                    break
            
            if not found_section and current_section:
                current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        risk_section = sections.get('危险分层', '')
        risk_info = self._extract_risk_info(risk_section)
        
        # Part 1: 患者可见
        patient_visible = {
            "diagnosis_top5": sections.get('诊断Top5', ''),
            "diagnosis_basis": sections.get('诊断依据', ''),
            "differential_diagnosis": sections.get('鉴别诊断', ''),
            "suggested_examinations": sections.get('建议的检查或检验', '')
        }
        
        # Part 2: 医生专属
        doctor_only = {
            "risk_level": risk_info.get('level', '未评估'),
            "risk_details": risk_section,
            "home_monitoring": risk_info.get('home_monitoring', ''),
            "visit_advice": risk_info.get('visit_advice', ''),
            "emergency_needed": risk_info.get('emergency_needed', ''),
            "response_procedure": risk_info.get('response_procedure', '')
        }
        
        return patient_visible, doctor_only
    
    def _extract_risk_info(self, risk_section: str) -> Dict[str, str]:
        """从危险分层部分提取关键信息"""
        info = {
            'level': '未评估',
            'home_monitoring': '',
            'visit_advice': '',
            'emergency_needed': '',
            'response_procedure': ''
        }
        
        if '急危' in risk_section or 'I级' in risk_section:
            info['level'] = '急危（I级）- 立即抢救'
        elif '急重' in risk_section or 'II级' in risk_section:
            info['level'] = '急重（II级）- 10分钟内救治'
        elif '急症' in risk_section or 'III级' in risk_section:
            info['level'] = '急症（III级）- 30分钟内接诊'
        elif '亚急症' in risk_section or 'IV级' in risk_section:
            info['level'] = '亚急症（IV级）- 60分钟内接诊'
        elif '非急症' in risk_section or 'V级' in risk_section:
            info['level'] = '非急症（V级）- 顺序就诊'
        
        # 提取各字段
        patterns = {
            'home_monitoring': [r'居家观[测察][：:](.*?)(?=就诊意见|是否需要急救|$)'],
            'visit_advice': [r'就诊意见[：:](.*?)(?=是否需要急救|居家|$)'],
            'emergency_needed': [r'是否需要急救[：:](.*?)(?=居家|就诊|$)'],
            'response_procedure': [r'响应程序[：:](.*?)(?=居家|就诊|是否|$)']
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, risk_section, re.DOTALL)
                if match:
                    info[field] = match.group(1).strip()
                    break
        
        return info
    
    def diagnose(
        self,
        patient_record: str,
        bowel_sound_result: Optional[Dict] = None,
        ecg_result: Optional[Dict] = None,
        max_experts: int = 5
    ) -> DiagnosisOutput:
        """
        主诊断接口
        
        Args:
            patient_record: 患者病历文本
            bowel_sound_result: 肠鸣音预测结果 {fold, pid, pred, prob_0, prob_1}
            ecg_result: ECG预测结果 {pid, path, pred_id, pred, conf, topk, probs_json}
            max_experts: 最大激活专家数量
        
        Returns:
            DiagnosisOutput: 包含患者可见和医生专属两部分的诊断结果
        """
        print("\n" + "="*60)
        print("开始诊断 - 28专家池")
        print("="*60)
        
        # 1. 整合患者信息
        patient_info = patient_record
        
        if bowel_sound_result:
            bs = BowelSoundResult.from_dict(bowel_sound_result)
            patient_info += "\n\n" + bs.to_clinical_text()
            print(f"✅ 已整合肠鸣音结果: pred={bs.pred}")
        
        if ecg_result:
            ecg = ECGResult.from_dict(ecg_result)
            patient_info += "\n\n" + ecg.to_clinical_text()
            print(f"✅ 已整合ECG结果: pred={ecg.pred}")
        
        # 2. Step 1: 科室路由
        step1_output, recommended_specialties = self._step1_route(patient_info)
        
        # 3. 从28专家池选择专家
        selected_experts = self._select_experts(recommended_specialties, max_experts)
        print(f"\n激活专家 ({len(selected_experts)}): {[e['name'] for e in selected_experts]}")
        
        # 4. Step 2 & 3: 专家诊断
        expert_opinions = []
        
        for idx, expert in enumerate(selected_experts):
            print(f"\n【Step 2&3】[{idx+1}/{len(selected_experts)}] {expert['name']} 分析中...")
            
            # Step 2: 语义重写
            rewritten = self._step2_rewrite(patient_info, expert)
            
            # Step 3: 诊断（使用evolved_prompt）
            diagnosis = self._step3_diagnose(patient_info, expert, rewritten)
            
            expert_opinions.append({
                "expert_id": expert['id'],
                "expert_name": expert['name'],
                "specialty": expert['specialty'],
                "fitness": expert.get('fitness', 0),
                "rewrite": rewritten,
                "diagnosis": diagnosis
            })
            
            print(f"  ✓ 完成")
        
        # 5. Step 4: 整合裁决
        step4_output = self._step4_aggregate(patient_info, expert_opinions)
        
        # 6. 解析输出
        patient_visible, doctor_only = self._parse_output(step4_output)
        
        print("\n" + "="*60)
        print("诊断完成！")
        print(f"危险分层: {doctor_only.get('risk_level', '未评估')}")
        print("="*60)
        
        return DiagnosisOutput(
            patient_visible=patient_visible,
            doctor_only=doctor_only,
            expert_opinions=expert_opinions,
            raw_output=step4_output
        )


# ============================================================
# 便捷接口函数
# ============================================================

def convert_patient_json_to_text(data: Dict[str, Any]) -> str:
    """
    将结构化患者JSON转换为文本提示
    
    支持两种输入格式:
    1. GPT输出格式: chiefComplaint, presentIllness, personalHistory, patientGender, patientAge
    2. 表格输入格式: gender (F/M), age, labs, exam, history, physical_examination
    """
    text_parts = []
    
    # 基本信息
    name = data.get('patientName', '')
    gender_raw = data.get('patientGender', data.get('gender', ''))
    age = data.get('patientAge', data.get('age', ''))
    
    # 性别映射: F/M -> 女/男
    gender_map = {'F': '女', 'M': '男', 'f': '女', 'm': '男'}
    gender = gender_map.get(gender_raw, gender_raw)  # 如果不是F/M则保持原值
    
    if name or gender or age:
        info = f"患者信息:"
        if name: info += f" 姓名: {name}"
        if gender: info += f" 性别: {gender}"
        if age: info += f" 年龄: {age}岁"
        text_parts.append(info)
    
    # 优先使用 GPT输出格式 (flat keys)
    if 'chiefComplaint' in data:
        text_parts.append(f"【主诉】\n{data.get('chiefComplaint', '')}")
    if 'presentIllness' in data:
        text_parts.append(f"【现病史】\n{data.get('presentIllness', '')}")
    if 'personalHistory' in data:
        text_parts.append(f"【既往史】\n{data.get('personalHistory', '')}")
        
    # 兼容 Input Table 格式 (nested objects)
    if 'history' in data and isinstance(data['history'], dict):
        hist = data['history']
        if 'Chief Complaint' in hist:
            text_parts.append(f"【主诉】\n{hist.get('Chief Complaint', '')}")
        if 'History of Present Illness' in hist:
            text_parts.append(f"【现病史】\n{hist.get('History of Present Illness', '')}")
    
    # 体格检查 (选填字段)
    if 'physical_examination' in data and data['physical_examination']:
        pe = data['physical_examination']
        if isinstance(pe, dict):
            pe_text = "【体格检查】\n"
            for k, v in pe.items():
                pe_text += f"{k}: {v}\n"
            text_parts.append(pe_text)
        else:
            text_parts.append(f"【体格检查】\n{pe}")
    
    # 检验指标 (labs)
    if 'labs' in data:
        labs = data['labs']
        if isinstance(labs, list) and labs:
            lab_text = "【检验指标】\n"
            for item in labs:
                if isinstance(item, dict):
                    key = item.get('key', '')
                    value = item.get('value', '')
                    lab_text += f"- {key}:\n  {value}\n"
            text_parts.append(lab_text.strip())
        elif labs:
            text_parts.append(f"【检验指标】\n{labs}")
            
    # 检查指标 (exam)
    if 'exam' in data:
        exam = data['exam']
        if isinstance(exam, list) and exam:
            exam_text = "【检查指标】\n"
            for item in exam:
                if isinstance(item, dict):
                    key = item.get('key', '')
                    value = item.get('value', '')
                    exam_text += f"- {key}:\n  {value}\n"
            text_parts.append(exam_text.strip())
        elif exam:
            text_parts.append(f"【检查指标】\n{exam}")

    # 如果都没有，尝试直接转储
    if not text_parts and data:
        return json.dumps(data, ensure_ascii=False, indent=2)
        
    return "\n\n".join(text_parts)


def diagnose_patient(
    patient_record: Any,
    bowel_sound_result: Optional[Dict] = None,
    ecg_result: Optional[Dict] = None,
    for_doctor: bool = False,
    max_experts: int = 5
) -> Dict[str, Any]:
    """
    便捷诊断接口
    
    Args:
        patient_record: 患者病历文本 或 结构化JSON(Dict)
        ...
    """
    # 处理输入格式
    if isinstance(patient_record, dict):
        patient_text = convert_patient_json_to_text(patient_record)
    else:
        patient_text = str(patient_record)

    api = DiagnosisAPI()
    output = api.diagnose(patient_text, bowel_sound_result, ecg_result, max_experts)
    
    if for_doctor:
        # 如果输入是字典，尝试提取上下文信息
        app_id = ""
        clinic_code = ""
        conversation_id = ""
        if isinstance(patient_record, dict):
            clinic_code = patient_record.get('clinicCode', '')
            # 其他字段如果存在于输入中可以提取
            
        return output.to_management_response(
            app_id=app_id,
            clinic_code=clinic_code,
            conversation_id=conversation_id
        )
    else:
        # 即使是患者端，现在也返回完整结构 (status, patient_info, risk_assessment...)
        # 复用 to_doctor_response 作为 diagnosis.json 的内容
        return output.to_doctor_response()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Diagnosis API")
    parser.add_argument("--patient_file", type=str, help="Path to patient JSON file")
    parser.add_argument("--c_file", type=str, help="Path to bowel sound JSON file")
    parser.add_argument("--ecg-file", type=str, help="Path to ECG JSON file")
    
    args = parser.parse_args()
    
    if args.patient_file:
        # CLI Mode
        print(f"Loading patient data from {args.patient_file}...")
        try:
            with open(args.patient_file, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
        except Exception as e:
            print(f"Error reading patient file: {e}")
            exit(1)
            
        bs_data = None
        if args.c_file and os.path.exists(args.c_file):
            print(f"Loading bowel sound data from {args.c_file}...")
            try:
                with open(args.c_file, 'r', encoding='utf-8') as f:
                    bs_data = json.load(f)
            except Exception as e:
                print(f"Error reading bowel sound file: {e}")

        ecg_data = None
        if args.ecg_file and os.path.exists(args.ecg_file):
            print(f"Loading ECG data from {args.ecg_file}...")
            try:
                with open(args.ecg_file, 'r', encoding='utf-8') as f:
                    ecg_data = json.load(f)
            except Exception as e:
                print(f"Error reading ECG file: {e}")

        # Execute Diagnosis
        # 转换患者数据为文本
        patient_text = convert_patient_json_to_text(patient_data)
        
        api = DiagnosisAPI()
        # Note: diagnose returns DiagnosisOutput object
        output = api.diagnose(patient_text, bs_data, ecg_data)
        
        # Output 1: diagnosis.json (Patient Interface + Risk)
        # Assuming diagnosis.json follows the structure of to_doctor_response (status, patient_info, risk...)
        diag_json = output.to_doctor_response()
        with open('diagnosis.json', 'w', encoding='utf-8') as f:
            json.dump(diag_json, f, ensure_ascii=False, indent=2)
        print("Generated diagnosis.json")
        
        # Output 2: doctor.json (Management Interface)
        clinic_code = patient_data.get('clinicCode', '')
        # Generate ids if not present
        doc_json = output.to_management_response(
            clinic_code=clinic_code
        )
        with open('doctor.json', 'w', encoding='utf-8') as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=2)
        print("Generated doctor.json")
        
    else:
        # Default Test Mode (Keep existing test for backward compatibility or debugging)
        sample_patient_record = """
患者ID: TEST_001
性别: 男
年龄: 52岁

【主诉】
腹痛伴恶心呕吐2天

【现病史】
患者2天前无明显诱因出现上腹部疼痛，呈阵发性绞痛，向右肩背部放射，
伴恶心呕吐，呕吐物为胃内容物，非喷射性。疼痛与进食油腻食物有关。
无发热，无黄疸，大小便正常。

【既往史】
高血压病史5年，规律服用降压药，血压控制良好。
无糖尿病、冠心病病史。

【体格检查】
T: 37.2℃, P: 88次/分, R: 18次/分, BP: 135/85mmHg
腹部：右上腹压痛（+），Murphy征阳性，无反跳痛，肝脾未触及

【实验室检查】
WBC: 11.2×10^9/L（偏高）
ALT: 45 U/L
AST: 38 U/L
总胆红素: 22 μmol/L（略高）
"""
        
        sample_bowel_sound = {
            "fold": "fold_0",
            "pid": "TEST_001",
            "pred": 0,
            "prob_0": 0.72,
            "prob_1": 0.28
        }
        
        sample_ecg = {
            "pid": "TEST_001",
            "path": "path/to/ecg.jpg",
            "pred_id": 0,
            "pred": False,
            "conf": 0.85,
            "topk": [["False", 0.85], ["True", 0.15]],
            "probs_json": '{"False": 0.85, "True": 0.15}'
        }
        
        print("="*60)
        print("诊断服务API测试 (使用 --patient_file 运行文件模式)")
        print("="*60)
        
        # 患者端
        print("\n【患者端响应】")
        patient_result = diagnose_patient(
            patient_record=sample_patient_record,
            bowel_sound_result=sample_bowel_sound,
            ecg_result=sample_ecg,
            for_doctor=False
        )
        print(json.dumps(patient_result, ensure_ascii=False, indent=2))
        
        # 医生端
        print("\n【医生端响应】")
        # Note: existing string input won't have clinicCode, so management response will be partial
        doctor_result = diagnose_patient(
            patient_record=sample_patient_record,
            bowel_sound_result=sample_bowel_sound,
            ecg_result=sample_ecg,
            for_doctor=True
        )
        print(json.dumps(doctor_result, ensure_ascii=False, indent=2))
