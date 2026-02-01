"""
Diagnosis Service API Interface

Input:
- Patient medical record text (integrated)
- Bowel sound prediction results (from other models)
- ECG prediction results (from other models)

Processing Flow:
- Step 1: Department Routing → Select corresponding experts from the pool
- Step 2: Expert Semantic Rewriting
- Step 3: Expert Diagnosis (using evolved_prompt from the pool)
- Step 4: Multi-expert Consensus Aggregation (including risk stratification, dual output)

Output:
- Part 1: Patient-visible diagnosis results (Top 5 diagnoses, basis, differential, suggestions)
- Part 2: Doctor-only risk stratification (Risk level, home monitoring, clinical advice, emergency need)
"""

import json
import re
import os
import argparse
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

from evomed.evoexperts.prompt.step1_route import system_step1_prompt
from evomed.evoexperts.prompt.step2_ir import system_step2_prompt
from evomed.evoexperts.prompt.step3_diag import system_step3_prompt
from evomed.evoexperts.prompt.step4_agg import system_step4_prompt as system_step4_prompt_v2

# =============================================================================
# API configuration
# =============================================================================
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

def get_default_expert_pool_path():
    env_path = os.environ.get("EXPERT_POOL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "optimized_expert_pool_28.json"),
        os.path.join(current_dir, "outputs", "optimized_expert_pool_28.json"),
        os.path.join(current_dir, "../outputs/optimized_expert_pool_28.json"),
        os.path.join(os.getcwd(), "optimized_expert_pool_28.json")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(current_dir, "../outputs/optimized_expert_pool_28.json")

EXPERT_POOL_28_PATH = get_default_expert_pool_path()


@dataclass
class BowelSoundResult:
    """Bowel Sound Prediction Result"""
    fold: str = ""
    pid: str = ""
    pred: int = 0  # 0 or 1
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
        """Converts to clinical description text"""
        pred_label = "Abnormal" if self.pred == 1 else "Normal"
        confidence = max(self.prob_0, self.prob_1) * 100
        
        text = f"[Bowel Sound Detection Results]\n"
        text += f"- Prediction: {pred_label}\n"
        text += f"- Confidence: {confidence:.1f}%\n"
        
        if self.pred == 1:
            text += f"- Abnormal Probability: {self.prob_1*100:.1f}%\n"
            text += f"- Note: Bowel sounds may be abnormal; clinical correlation suggested.\n"
        else:
            text += f"- Normal Probability: {self.prob_0*100:.1f}%\n"
            text += f"- Note: No obvious abnormality in bowel sounds.\n"
        
        return text


@dataclass
class ECGResult:
    """ECG Prediction Result"""
    pid: str = ""
    path: str = ""
    pred_id: int = 0
    pred: bool = False  # False=Normal, True=Abnormal
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
        """Converts to clinical description text"""
        pred_label = "Abnormal" if self.pred else "Normal"
        confidence = self.conf * 100
        
        text = f"[ECG AI Analysis Results]\n"
        text += f"- AI Prediction: {pred_label}\n"
        text += f"- Confidence: {confidence:.1f}%\n"
        
        if self.pred:
            text += f"- Note: ECG may be abnormal; professional review and clinical correlation suggested.\n"
        else:
            text += f"- Note: No obvious abnormality in AI analysis of ECG.\n"
        
        if self.topk:
            text += f"- Detailed Probabilities:\n"
            for label, prob in self.topk:
                label_en = "Abnormal" if label == "True" else "Normal"
                text += f"    {label_en}: {prob*100:.1f}%\n"
        
        return text


@dataclass
class DiagnosisOutput:
    """Diagnosis Output Results"""
    patient_visible: Dict[str, Any] = field(default_factory=dict)
    doctor_only: Dict[str, Any] = field(default_factory=dict)
    expert_opinions: List[Dict] = field(default_factory=list)
    raw_output: str = ""
    
    def to_patient_response(self) -> Dict[str, Any]:
        """Returns content visible to patient"""
        return {
            "status": "success",
            "data": self.patient_visible,
            "notice": "The above is AI-assisted diagnostic advice for reference only. Please consult a professional doctor."
        }
    
    def to_doctor_response(self) -> Dict[str, Any]:
        """Returns full content including doctor-only part"""
        return {
            "status": "success",
            "patient_info": self.patient_visible,
            "risk_assessment": self.doctor_only,
            "expert_opinions": self.expert_opinions,
            "raw_output": self.raw_output
        }

    def to_management_response(self, app_id: str = "", conversation_id: str = "", clinic_code: str = "") -> Dict[str, Any]:
        """Returns management/doctor-side format"""
        risk_level = self.doctor_only.get('risk_level', '')
        top1_diagnosis = "Unknown"
        if self.patient_visible.get('diagnosis_top5'):
             lines = self.patient_visible['diagnosis_top5'].split('\n')
             for line in lines:
                 if line.strip():
                     top1_diagnosis = line.strip()
                     break
        
        summary = f"Risk Level: {risk_level}. Initial Diagnosis: {top1_diagnosis}. Clinical follow-up suggested."
        diag_result = self.patient_visible.get('diagnosis_top5', '')
        
        condition_analysis = ""
        if self.patient_visible.get('diagnosis_basis'):
             condition_analysis += f"[Diagnostic Basis]\n{self.patient_visible['diagnosis_basis']}\n\n"
        if self.patient_visible.get('differential_diagnosis'):
             condition_analysis += f"[Differential Diagnosis]\n{self.patient_visible['differential_diagnosis']}"
        
        suggestions = self.patient_visible.get('suggested_examinations', '')

        return {
            "type": "DiagnosticMessage",
            "app_id": app_id or str(uuid.uuid4()),
            "summary_label": "Summary",
            "summary_value": summary,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "clinic_code": clinic_code, 
            "diagnostic_result_label": "Diagnosis",
            "diagnostic_result_value": diag_result,
            "condition_analysis_label": "Analysis",
            "condition_analysis_value": condition_analysis.strip(),
            "suggestions_examinations_label": "Next Steps",
            "suggestions_examinations_value": suggestions,
            "diffDiagnosis_and_conditionAnalysis_label": "Analysis",
            "diffDiagnosis_and_conditionAnalysis_value": condition_analysis.strip()
        }


class DiagnosisAPI:
    """
    Diagnosis Service API
    """
    
    def __init__(self, expert_pool_path: str = EXPERT_POOL_28_PATH):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
        
        self.resource = """[Medical Resources]
Available Departments: Internal Medicine, Surgery, Emergency, Cardiology, Gastroenterology, Respiratory, Neurology, Oncology, Orthopedics, Urology, Hepatobiliary Surgery, OB/GYN, Pediatrics, etc.
Knowledge Bases: Medical Guidelines, Clinical Experience Library, Similar Case Library.
Note: This system provides diagnostic assistance only and does not replace professional medical activities."""
        
        print("="*60)
        print("Initializing Diagnosis API - 28 Expert Pool")
        print("="*60)
        
        self.expert_pool_28 = self._load_expert_pool(expert_pool_path)
        if self.expert_pool_28:
            print(f"✅ Successfully loaded 28 expert pool: {len(self.expert_pool_28)} experts")
            specialties = set(e.get('specialty', '') for e in self.expert_pool_28)
            print(f"   Covered Specialties: {len(specialties)}")
        else:
            raise RuntimeError(f"Could not load expert pool: {expert_pool_path}")
        
        self.specialty_to_experts = self._build_specialty_mapping()
    
    def _load_expert_pool(self, path: str) -> Optional[List[Dict]]:
        if not os.path.exists(path):
            print(f"  ❌ Expert pool file does not exist: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            for expert in pool:
                if 'evolved_prompt' in expert and 'prompt' not in expert:
                    expert['prompt'] = expert['evolved_prompt']
            return pool
        except Exception as e:
            print(f"  ❌ Failed to load expert pool: {e}")
            return None
    
    def _build_specialty_mapping(self) -> Dict[str, List[Dict]]:
        mapping = {}
        for expert in self.expert_pool_28:
            specialty = expert.get('specialty', '')
            if specialty:
                if specialty not in mapping:
                    mapping[specialty] = []
                mapping[specialty].append(expert)
        
        for specialty in mapping:
            mapping[specialty] = sorted(
                mapping[specialty], 
                key=lambda x: x.get('fitness', 0), 
                reverse=True
            )
        
        aliases = {
            "Obstetrics and Gynecology": ["OB/GYN", "Gynecology", "Obstetrics"],
            "Gastroenterology": ["Gastro", "GI"],
            "Cardiology": ["Heart", "Cardiac"],
            "Respiratory Medicine": ["Pulmonology", "Lung"],
            "Hepatobiliary Surgery": ["Hepatobiliary"],
            "Gastrointestinal Surgery": ["GI Surgery", "General Surgery"],
            "Urology": ["Uro"],
            "General Practice": ["GP"],
        }
        
        for main_specialty, alias_list in aliases.items():
            if main_specialty in mapping:
                for alias in alias_list:
                    mapping[alias] = mapping[main_specialty]
        
        return mapping
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        import time
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a professional medical diagnostic assistant agent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def _step1_route(self, patient_info: str) -> Tuple[str, List[str]]:
        print("\n[Step 1] Department Routing...")
        available_specialties = list(set(e.get('specialty', '') for e in self.expert_pool_28))
        prompt = system_step1_prompt.format(
            resource=self.resource + f"\n\n[Optional Specialties]\n{', '.join(available_specialties)}",
            patient=patient_info
        )
        result = self._call_llm(prompt)
        recommended = self._extract_specialties_from_pool(result)
        print(f"  Recommended: {recommended}")
        return result, recommended
    
    def _extract_specialties_from_pool(self, step1_output: str) -> List[str]:
        pool_specialties = set(e.get('specialty', '') for e in self.expert_pool_28)
        alias_map = {
            "Cardiology": "Cardiology",
            "Gastro": "Gastroenterology",
            "GI": "Gastroenterology",
            "GP": "General Practice",
            "OB/GYN": "Obstetrics and Gynecology",
        }
        
        found = []
        for keyword in list(pool_specialties) + list(alias_map.keys()):
            if keyword.lower() in step1_output.lower():
                normalized = alias_map.get(keyword, keyword)
                if normalized in pool_specialties and normalized not in found:
                    found.append(normalized)
        
        if not found:
            found = ["General Practice"]
        return found
    
    def _select_experts(self, recommended_specialties: List[str], max_experts: int = 5) -> List[Dict]:
        selected = []
        selected_ids = set()
        for specialty in recommended_specialties:
            if len(selected) >= max_experts:
                break
            experts = self.specialty_to_experts.get(specialty, [])
            for expert in experts:
                if expert['id'] not in selected_ids:
                    selected.append(expert)
                    selected_ids.add(expert['id'])
                    print(f"  Selected: {expert['name']} (fitness={expert.get('fitness', 0):.3f})")
                    break
        
        if len(selected) < 2:
            sorted_pool = sorted(self.expert_pool_28, key=lambda x: x.get('fitness', 0), reverse=True)
            for expert in sorted_pool:
                if expert['id'] not in selected_ids:
                    selected.append(expert)
                    selected_ids.add(expert['id'])
                    print(f"  Supplemented: {expert['name']} (fitness={expert.get('fitness', 0):.3f})")
                    if len(selected) >= 3:
                        break
        return selected
    
    def _step2_rewrite(self, patient_info: str, expert: Dict) -> str:
        expert_desc = f"""
Expert Role: {expert.get('name', 'Expert')}
Specialty: {expert.get('specialty', '')}
Professional Description: {expert.get('description', '')}
Focus Areas: {', '.join(expert.get('focus_areas', []))}
"""
        prompt = system_step2_prompt.format(
            resource=self.resource,
            patient=patient_info,
            expert=expert_desc
        )
        return self._call_llm(prompt)
    
    def _step3_diagnose(self, patient_info: str, expert: Dict, rewritten_info: str) -> str:
        expert_desc = f"""
Expert Role: {expert.get('name', 'Expert')}
Specialty: {expert.get('specialty', '')}
Professional Description: {expert.get('description', '')}
Focus Areas: {', '.join(expert.get('focus_areas', []))}

[Rewritten Patient Info from Expert's Perspective]
{rewritten_info}
"""
        template = expert.get('prompt') or expert.get('evolved_prompt') or system_step3_prompt
        prompt = template.format(
            resource=self.resource,
            patient=patient_info,
            expert=expert_desc,
            reference="[Reference Materials] None"
        )
        return self._call_llm(prompt)
    
    def _step4_aggregate(self, patient_info: str, expert_opinions: List[Dict]) -> str:
        print("\n[Step 4] Multi-Expert Aggregation...")
        experts_summary = ""
        for opinion in expert_opinions:
            experts_summary += f"""
{'='*40}
Opinion from [{opinion['expert_name']}]
Specialty: {opinion['specialty']}
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
        sections = {
            'Risk Stratification': '',
            'Diagnosis Top 5': '',
            'Diagnostic Basis': '',
            'Differential Diagnosis': '',
            'Suggested Examinations or Tests': ''
        }
        
        current_section = None
        lines = raw_output.split('\n')
        current_content = []
        
        for line in lines:
            found_section = False
            for section_name in sections.keys():
                if section_name.lower() in line.lower() and any(m in line for m in ['I.', 'II.', 'III.', 'IV.', 'V.']):
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
        
        risk_section = sections.get('Risk Stratification', '')
        risk_info = self._extract_risk_info(risk_section)
        
        patient_visible = {
            "diagnosis_top5": sections.get('Diagnosis Top 5', ''),
            "diagnosis_basis": sections.get('Diagnostic Basis', ''),
            "differential_diagnosis": sections.get('Differential Diagnosis', ''),
            "suggested_examinations": sections.get('Suggested Examinations or Tests', '')
        }
        
        doctor_only = {
            "risk_level": risk_info.get('level', 'Not Evaluated'),
            "risk_details": risk_section,
            "home_monitoring": risk_info.get('home_monitoring', ''),
            "visit_advice": risk_info.get('visit_advice', ''),
            "emergency_needed": risk_info.get('emergency_needed', ''),
            "response_procedure": risk_info.get('response_procedure', '')
        }
        
        return patient_visible, doctor_only
    
    def _extract_risk_info(self, risk_section: str) -> Dict[str, str]:
        info = {
            'level': 'Not Evaluated',
            'home_monitoring': '',
            'visit_advice': '',
            'emergency_needed': '',
            'response_procedure': ''
        }
        
        low_rs = risk_section.lower()
        if 'critical' in low_rs or 'level i' in low_rs:
            info['level'] = 'Critical (Level I) - Immediate Resuscitation'
        elif 'emergency' in low_rs or 'level ii' in low_rs:
            info['level'] = 'Emergency (Level II) - Care within 10 mins'
        elif 'urgent' in low_rs or 'level iii' in low_rs:
            info['level'] = 'Urgent (Level III) - Seen within 30 mins'
        elif 'semi-urgent' in low_rs or 'level iv' in low_rs:
            info['level'] = 'Semi-Urgent (Level IV) - Seen within 60 mins'
        elif 'non-urgent' in low_rs or 'level v' in low_rs:
            info['level'] = 'Non-Urgent (Level V) - Sequential visit'
        
        patterns = {
            'home_monitoring': [r'Home Monitoring[：:](.*?)(?=Visit Advice|Emergency Needed|$)'],
            'visit_advice': [r'Visit Advice[：:](.*?)(?=Emergency Needed|Home|$)'],
            'emergency_needed': [r'Emergency Needed[：:](.*?)(?=Home|Visit|$)'],
            'response_procedure': [r'Response Procedure[：:](.*?)(?=Home|Visit|Emergency|$)']
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, risk_section, re.IGNORECASE | re.DOTALL)
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
        print("\n" + "="*60)
        print("Starting Diagnosis - 28 Expert Pool")
        print("="*60)
        
        patient_info = patient_record
        if bowel_sound_result:
            bs = BowelSoundResult.from_dict(bowel_sound_result)
            patient_info += "\n\n" + bs.to_clinical_text()
        if ecg_result:
            ecg = ECGResult.from_dict(ecg_result)
            patient_info += "\n\n" + ecg.to_clinical_text()
        
        step1_output, recommended_specialties = self._step1_route(patient_info)
        selected_experts = self._select_experts(recommended_specialties, max_experts)
        
        expert_opinions = []
        for idx, expert in enumerate(selected_experts):
            print(f"\n[Step 2&3] [{idx+1}/{len(selected_experts)}] {expert['name']} Analyzing...")
            rewritten = self._step2_rewrite(patient_info, expert)
            diagnosis = self._step3_diagnose(patient_info, expert, rewritten)
            expert_opinions.append({
                "expert_id": expert['id'],
                "expert_name": expert['name'],
                "specialty": expert['specialty'],
                "fitness": expert.get('fitness', 0),
                "rewrite": rewritten,
                "diagnosis": diagnosis
            })
        
        step4_output = self._step4_aggregate(patient_info, expert_opinions)
        patient_visible, doctor_only = self._parse_output(step4_output)
        
        print("\n" + "="*60)
        print("Diagnosis Complete!")
        print(f"Risk Level: {doctor_only.get('risk_level', 'Not Evaluated')}")
        print("="*60)
        
        return DiagnosisOutput(
            patient_visible=patient_visible,
            doctor_only=doctor_only,
            expert_opinions=expert_opinions,
            raw_output=step4_output
        )


def convert_patient_json_to_text(data: Dict[str, Any]) -> str:
    text_parts = []
    name = data.get('patientName', '')
    gender_raw = data.get('patientGender', data.get('gender', ''))
    age = data.get('patientAge', data.get('age', ''))
    
    gender_map = {'F': 'Female', 'M': 'Male', 'f': 'Female', 'm': 'Male'}
    gender = gender_map.get(gender_raw, gender_raw)
    
    if name or gender or age:
        info = f"Patient Info:"
        if name: info += f" Name: {name}"
        if gender: info += f" Gender: {gender}"
        if age: info += f" Age: {age} years"
        text_parts.append(info)
    
    if 'chiefComplaint' in data:
        text_parts.append(f"[Chief Complaint]\n{data.get('chiefComplaint', '')}")
    if 'presentIllness' in data:
        text_parts.append(f"[History of Present Illness]\n{data.get('presentIllness', '')}")
    if 'personalHistory' in data:
        text_parts.append(f"[Past History]\n{data.get('personalHistory', '')}")
        
    if 'history' in data and isinstance(data['history'], dict):
        hist = data['history']
        if 'Chief Complaint' in hist:
            text_parts.append(f"[Chief Complaint]\n{hist.get('Chief Complaint', '')}")
        if 'History of Present Illness' in hist:
            text_parts.append(f"[History of Present Illness]\n{hist.get('History of Present Illness', '')}")
    
    if 'physical_examination' in data and data['physical_examination']:
        pe = data['physical_examination']
        if isinstance(pe, dict):
            pe_text = "[Physical Examination]\n"
            for k, v in pe.items():
                pe_text += f"{k}: {v}\n"
            text_parts.append(pe_text)
        else:
            text_parts.append(f"[Physical Examination]\n{pe}")
    
    if 'labs' in data:
        labs = data['labs']
        if isinstance(labs, list) and labs:
            lab_text = "[Laboratory Results]\n"
            for item in labs:
                if isinstance(item, dict):
                    key = item.get('key', '')
                    value = item.get('value', '')
                    lab_text += f"- {key}:\n  {value}\n"
            text_parts.append(lab_text.strip())
        elif labs:
            text_parts.append(f"[Laboratory Results]\n{labs}")
            
    if 'exam' in data:
        exam = data['exam']
        if isinstance(exam, list) and exam:
            exam_text = "[Imaging Results]\n"
            for item in exam:
                if isinstance(item, dict):
                    key = item.get('key', '')
                    value = item.get('value', '')
                    exam_text += f"- {key}:\n  {value}\n"
            text_parts.append(exam_text.strip())
        elif exam:
            text_parts.append(f"[Imaging Results]\n{exam}")

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
    if isinstance(patient_record, dict):
        patient_text = convert_patient_json_to_text(patient_record)
    else:
        patient_text = str(patient_record)

    api = DiagnosisAPI()
    output = api.diagnose(patient_text, bowel_sound_result, ecg_result, max_experts)
    
    if for_doctor:
        app_id = ""
        clinic_code = ""
        conversation_id = ""
        if isinstance(patient_record, dict):
            clinic_code = patient_record.get('clinicCode', '')
        return output.to_management_response(
            app_id=app_id,
            clinic_code=clinic_code,
            conversation_id=conversation_id
        )
    else:
        return output.to_doctor_response()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Diagnosis API")
    parser.add_argument("--patient_file", type=str, help="Path to patient JSON file")
    parser.add_argument("--c_file", type=str, help="Path to bowel sound JSON file")
    parser.add_argument("--ecg-file", type=str, help="Path to ECG JSON file")
    args = parser.parse_args()
    
    if args.patient_file:
        print(f"Loading patient data from {args.patient_file}...")
        try:
            with open(args.patient_file, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
        except Exception as e:
            print(f"Error reading patient file: {e}")
            exit(1)
            
        bs_data = None
        if args.c_file and os.path.exists(args.c_file):
            try:
                with open(args.c_file, 'r', encoding='utf-8') as f:
                    bs_data = json.load(f)
            except: pass

        ecg_data = None
        if args.ecg_file and os.path.exists(args.ecg_file):
            try:
                with open(args.ecg_file, 'r', encoding='utf-8') as f:
                    ecg_data = json.load(f)
            except: pass

        patient_text = convert_patient_json_to_text(patient_data)
        api = DiagnosisAPI()
        output = api.diagnose(patient_text, bs_data, ecg_data)
        
        diag_json = output.to_doctor_response()
        with open('diagnosis.json', 'w', encoding='utf-8') as f:
            json.dump(diag_json, f, indent=2)
        
        clinic_code = patient_data.get('clinicCode', '')
        doc_json = output.to_management_response(clinic_code=clinic_code)
        with open('doctor.json', 'w', encoding='utf-8') as f:
            json.dump(doc_json, f, indent=2)
    else:
        sample_patient_record = """
Patient ID: TEST_001
Gender: Male
Age: 52 years old

[Chief Complaint]
Abdominal pain with nausea and vomiting for 2 days

[History of Present Illness]
The patient developed upper abdominal pain without obvious triggers 2 days ago, presenting as paroxysmal colic radiating to the right shoulder and back, accompanied by nausea and vomiting (gastric content, non-projectile). Pain is related to fatty food intake. No fever, no jaundice, normal bowel movements.

[Past History]
Hypertension for 5 years, well-controlled with medication. No history of diabetes or coronary heart disease.

[Physical Examination]
T: 37.2℃, P: 88bpm, R: 18bpm, BP: 135/85mmHg
Abdomen: RUQ tenderness (+), Murphy sign positive, no rebound tenderness, liver/spleen not palpable.

[Laboratory Results]
WBC: 11.2×10^9/L (high)
ALT: 45 U/L
AST: 38 U/L
Total Bilirubin: 22 μmol/L (slightly high)
"""
        print("="*60)
        print("Diagnosis API Test")
        print("="*60)
        
        patient_result = diagnose_patient(patient_record=sample_patient_record)
        print(json.dumps(patient_result, indent=2))
