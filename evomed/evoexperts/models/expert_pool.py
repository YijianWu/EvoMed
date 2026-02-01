import os
"""
Evolving Expert Pool (EEP)

Supports:
1. Formal representation of expert units (ExpertUnit)
2. Episode-conditioned expert activation
3. Incremental construction and evolution of the expert pool
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import hashlib


# API Configuration
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass
class ExpertMetadata:
    """Structured Metadata for Experts (M_k)"""
    specialty_tags: List[str]           # Specialty tags
    applicable_stages: List[str]        # Applicable diagnostic stages: ["Initial", "Follow-up", "Inpatient", "Emergency"]
    risk_preference: str                # Risk preference: "Conservative", "Neutral", "Aggressive"
    target_populations: List[str]       # Typical target populations: ["Adult", "Elderly", "Child", "Maternal"]
    system_focus: List[str]             # Focused systems: ["Digestive", "Cardiovascular", ...]
    confidence_domains: List[str]       # High-confidence domains
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0                # Usage statistics
    effectiveness_score: float = 0.5    # Effectiveness score [0, 1]


@dataclass
class ExpertUnit:
    """
    Expert Unit E_k = (D_k, M_k, V_k)
    
    D_k: Expert background and perspective description (Natural Language)
    M_k: Structured metadata
    V_k: Vector representation (for semantic matching)
    """
    id: str                             # Unique identifier
    name: str                           # Expert name
    specialty: str                      # Main specialty
    description: str                    # D_k: Background and perspective description
    focus_areas: List[str]              # Focus areas
    thinking_patterns: List[str]        # Typical thinking paths
    metadata: ExpertMetadata            # M_k: Structured metadata
    embedding: Optional[List[float]] = None  # V_k: Vector representation
    
    def to_dict(self) -> Dict:
        """Converts to dictionary (for serialization)"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertUnit':
        """Creates from dictionary (for deserialization)"""
        metadata = ExpertMetadata(**data.pop('metadata'))
        return cls(metadata=metadata, **data)
    
    def get_description_for_prompt(self) -> str:
        """Generates expert description for use in prompts"""
        return f"""
Expert Role: {self.name}
Specialty: {self.specialty}
Professional Description: {self.description}
Focus Areas: {', '.join(self.focus_areas)}
Thinking Patterns: {', '.join(self.thinking_patterns)}
Risk Preference: {self.metadata.risk_preference}
Target Populations: {', '.join(self.metadata.target_populations)}
"""


@dataclass
class DiagnosticEpisode:
    """
    Diagnostic Episode ε = (S_ε, K_ε, T_ε)
    
    S_ε: Summary of minimum sufficient evidence
    K_ε: Set of keywords
    T_ε: Set of tags (Symptom domain, system domain, risk domain, etc.)
    """
    episode_id: str
    summary: str                        # S_ε: Evidence summary
    keywords: Set[str]                  # K_ε: Keyword set
    tags: Dict[str, List[str]]          # T_ε: Tag set
    stage: str = "Initial"              # Diagnostic stage
    embedding: Optional[List[float]] = None  # Vector representation
    
    @classmethod
    def from_patient_info(cls, patient_info: Any, episode_id: str = None) -> 'DiagnosticEpisode':
        """Creates an Episode from patient information"""
        if episode_id is None:
            episode_id = hashlib.md5(patient_info.patient_id.encode()).hexdigest()[:12]
        
        # Build evidence summary
        summary = f"""
Patient: {patient_info.gender}, {patient_info.age} years old
Chief Complaint: {patient_info.chief_complaint}
History of Present Illness: {patient_info.history_of_present_illness[:500] if patient_info.history_of_present_illness else 'None'}
Past History: {patient_info.past_history[:300] if patient_info.past_history else 'None'}
"""
        
        # Extract keywords (simplified)
        keywords = set()
        text = f"{patient_info.chief_complaint} {patient_info.history_of_present_illness}"
        # Common symptom keywords (English)
        symptom_keywords = ["pain", "abdominal pain", "abdominal distension", "headache", "chest tightness", "fever", "cough", 
                          "nausea", "vomiting", "diarrhea", "constipation", "bleeding", "edema", "fatigue"]
        for kw in symptom_keywords:
            if kw in text.lower():
                keywords.add(kw)
        
        # Build tags
        tags = {
            "symptom_domain": [],
            "system_domain": [],
            "risk_domain": [],
            "population": []
        }
        
        # System domain inference
        system_keywords = {
            "Digestive System": ["abdominal pain", "abdominal distension", "nausea", "vomiting", "diarrhea", "constipation", "stomach", "intestine"],
            "Cardiovascular System": ["chest tightness", "palpitations", "heart", "blood pressure", "heart failure", "atrial fibrillation"],
            "Respiratory System": ["cough", "sputum", "shortness of breath", "dyspnea", "lung"],
            "Urinary System": ["frequent urination", "urgent urination", "painful urination", "kidney", "bladder"],
            "Neurological System": ["headache", "dizziness", "consciousness", "numbness", "brain"],
            "Endocrine System": ["diabetes", "thyroid", "blood sugar"],
            "Reproductive System": ["menstruation", "pregnancy", "uterus", "ovary"],
        }
        
        for system, kws in system_keywords.items():
            for kw in kws:
                if kw in text.lower():
                    if system not in tags["system_domain"]:
                        tags["system_domain"].append(system)
                    break
        
        # Population characteristics
        if patient_info.age < 18:
            tags["population"].append("Child")
        elif patient_info.age >= 65:
            tags["population"].append("Elderly")
        else:
            tags["population"].append("Adult")
            
        if patient_info.gender == "Female":
            tags["population"].append("Female")
        else:
            tags["population"].append("Male")
        
        return cls(
            episode_id=episode_id,
            summary=summary,
            keywords=keywords,
            tags=tags,
            stage="Initial"
        )


class EvolvingExpertPool:
    """
    Evolving Expert Pool (EEP)
    """
    
    def __init__(self, pool_path: str = "expert_pool.json"):
        self.pool_path = Path(pool_path)
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        self.experts: Dict[str, ExpertUnit] = {}
        self._load_pool()
    
    def _load_pool(self):
        """Loads the expert resource pool"""
        if self.pool_path.exists():
            with open(self.pool_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for expert_data in data.get('experts', []):
                    expert = ExpertUnit.from_dict(expert_data)
                    self.experts[expert.id] = expert
            print(f"Loaded {len(self.experts)} expert units")
        else:
            self._initialize_default_experts()
            self._save_pool()
    
    def _save_pool(self):
        """Saves the expert resource pool"""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'experts': [e.to_dict() for e in self.experts.values()]
        }
        with open(self.pool_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Gets vector representation of text"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return [0.0] * 1536
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculates cosine similarity"""
        v1, v2 = np.array(v1), np.array(v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _initialize_default_experts(self):
        """Initializes default expert units"""
        default_experts = [
            ExpertUnit(
                id="obgyn",
                name="OB/GYN Expert",
                specialty="Obstetrics and Gynecology",
                description="Specializes in the diagnosis and treatment of female reproductive system diseases and perinatal care, focusing on women's health throughout their life cycle.",
                focus_areas=["Perinatal Management", "Gynecological Oncology", "Menstrual Disorders", "Reproductive Tract Inflammation"],
                thinking_patterns=["Reproductive endocrine axis assessment", "Pregnancy-related symptom differentiation", "Pelvic mass analysis"],
                metadata=ExpertMetadata(
                    specialty_tags=["OB/GYN", "Gynecology", "Obstetrics"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Female", "Maternal"],
                    system_focus=["Reproductive System"],
                    confidence_domains=["Gynecological Oncology", "AUB", "Pregnancy Complications"]
                )
            ),
            ExpertUnit(
                id="gastroenterology",
                name="Gastroenterology Expert",
                specialty="Gastroenterology",
                description="Specializes in the medical diagnosis and treatment of esophagus, stomach, intestine, liver, gallbladder, and pancreas diseases and endoscopic examination.",
                focus_areas=["Gastritis and Ulcers", "Liver Diseases", "Functional Gastrointestinal Disorders", "Gastrointestinal Bleeding"],
                thinking_patterns=["Digestive tract symptom localization", "Liver function abnormality analysis", "Endoscopic indication assessment"],
                metadata=ExpertMetadata(
                    specialty_tags=["Gastroenterology", "GI", "Digestive"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Digestive System"],
                    confidence_domains=["Peptic Ulcer", "Liver Cirrhosis", "IBD"]
                )
            ),
            ExpertUnit(
                id="pediatrics",
                name="Pediatrics Expert",
                specialty="Pediatrics",
                description="Focuses on growth, development, and disease diagnosis and treatment from newborn to adolescence, paying attention to children's unique physiological and pathological characteristics.",
                focus_areas=["Respiratory Infections", "Growth Assessment", "Pediatric Digestive System", "Newborn Care"],
                thinking_patterns=["Age-stratified assessment", "Developmental milestone comparison", "Pediatric dosage calculation"],
                metadata=ExpertMetadata(
                    specialty_tags=["Pediatrics", "Peds"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient", "Emergency"],
                    risk_preference="Conservative",
                    target_populations=["Child", "Newborn", "Adolescent"],
                    system_focus=["Respiratory", "Digestive", "Neurological"],
                    confidence_domains=["Pediatric Respiratory Infections", "Developmental Delay", "Pediatric Diarrhea"]
                )
            ),
            ExpertUnit(
                id="endocrinology",
                name="Endocrinology Expert",
                specialty="Endocrinology",
                description="Specializes in the diagnosis and long-term management of hormone secretion abnormalities and metabolic diseases.",
                focus_areas=["Diabetes Management", "Thyroid Diseases", "Osteoporosis", "Obesity and Metabolic Syndrome"],
                thinking_patterns=["Hormonal axis function assessment", "Metabolic index comprehensive analysis", "Long-term complication prevention"],
                metadata=ExpertMetadata(
                    specialty_tags=["Endocrinology", "Metabolism"],
                    applicable_stages=["Initial", "Follow-up"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Endocrine System"],
                    confidence_domains=["Diabetes", "Thyroid Abnormality", "Bone Metabolism Diseases"]
                )
            ),
            ExpertUnit(
                id="hepatobiliary_surgery",
                name="Hepatobiliary Surgery Expert",
                specialty="Hepatobiliary Surgery",
                description="Specializes in surgical treatment and perioperative management of liver, biliary tract, and pancreatic diseases.",
                focus_areas=["Cholelithiasis", "Liver Tumors", "Pancreatitis", "Biliary Obstruction"],
                thinking_patterns=["Hepatobiliary imaging interpretation", "Surgical indication assessment", "Perioperative risk stratification"],
                metadata=ExpertMetadata(
                    specialty_tags=["Hepatobiliary Surgery", "Liver Surgery"],
                    applicable_stages=["Follow-up", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Digestive System"],
                    confidence_domains=["Gallbladder Stones", "Liver Cancer", "Pancreatic Tumor"]
                )
            ),
            ExpertUnit(
                id="orthopedics",
                name="Orthopedics Expert",
                specialty="Orthopedics",
                description="Specializes in diagnosis, reduction, and surgical treatment of motor system diseases such as bones, joints, muscles, and ligaments.",
                focus_areas=["Fractures/Trauma", "Arthritis", "Spine Diseases", "Sports Injuries"],
                thinking_patterns=["Fracture classification assessment", "Joint function score", "Surgical vs conservative treatment decision"],
                metadata=ExpertMetadata(
                    specialty_tags=["Orthopedics", "Ortho", "Traumatology"],
                    applicable_stages=["Initial", "Emergency", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly", "Child"],
                    system_focus=["Motor System"],
                    confidence_domains=["Extremity Fractures", "Degenerative Joint Disease", "Lumbar Disc Herniation"]
                )
            ),
            ExpertUnit(
                id="respiratory",
                name="Respiratory Expert",
                specialty="Respiratory Medicine",
                description="Specializes in medical diagnosis and treatment of respiratory system infections, airway diseases, and lung tumors.",
                focus_areas=["COPD", "Asthma Management", "Lung Nodules", "Pulmonary Infections"],
                thinking_patterns=["Lung function assessment", "Imaging sign interpretation", "Infection vs tumor differentiation"],
                metadata=ExpertMetadata(
                    specialty_tags=["Respiratory Medicine", "Pulmonology"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient", "Emergency"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Respiratory System"],
                    confidence_domains=["Pneumonia", "COPD", "Lung Cancer Screening"]
                )
            ),
            ExpertUnit(
                id="emergency",
                name="Emergency Expert",
                specialty="Emergency Medicine",
                description="Specializes in initial assessment, emergency resuscitation, and triage of acute illnesses, trauma, and various critical conditions.",
                focus_areas=["Vital Signs Maintenance", "Acute Poisoning", "Multiple Trauma", "CPR"],
                thinking_patterns=["ABCDE assessment", "Critical illness recognition", "Rapid triage decision"],
                metadata=ExpertMetadata(
                    specialty_tags=["Emergency Medicine", "ER"],
                    applicable_stages=["Emergency"],
                    risk_preference="Aggressive",
                    target_populations=["Adult", "Elderly", "Child"],
                    system_focus=["Cardiovascular", "Respiratory", "Neurological"],
                    confidence_domains=["Acute Chest Pain", "Shock", "Acute Abdominal Pain"]
                )
            ),
            ExpertUnit(
                id="urology",
                name="Urology Expert",
                specialty="Urology",
                description="Specializes in surgical and minimally invasive treatment of urinary system and male reproductive system diseases.",
                focus_areas=["Urinary Calculi", "Prostate Diseases", "Urinary Tract Tumors", "Urinary Tract Infections"],
                thinking_patterns=["Urinary tract obstruction assessment", "PSA interpretation", "Minimally invasive vs open surgery choice"],
                metadata=ExpertMetadata(
                    specialty_tags=["Urology", "Uro"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly", "Male"],
                    system_focus=["Urinary System", "Reproductive System"],
                    confidence_domains=["Kidney Stones", "BPH", "Bladder Cancer"]
                )
            ),
            ExpertUnit(
                id="nephrology",
                name="Nephrology Expert",
                specialty="Nephrology",
                description="Specializes in medical diagnosis and treatment of kidney diseases, including renal insufficiency, nephritis, and dialysis management.",
                focus_areas=["Chronic Kidney Disease Management", "Nephritic Syndrome", "Dialysis Treatment", "Electrolyte Imbalance"],
                thinking_patterns=["Renal function stage assessment", "Proteinuria analysis", "Renal replacement therapy decision"],
                metadata=ExpertMetadata(
                    specialty_tags=["Nephrology", "Renal"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient"],
                    risk_preference="Conservative",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Urinary System"],
                    confidence_domains=["CKD", "Diabetic Nephropathy", "IgA Nephropathy"]
                )
            ),
            ExpertUnit(
                id="general_practice",
                name="General Practice Expert",
                specialty="General Practice",
                description="Provides comprehensive and continuous basic medical services, specializing in initial diagnosis of undifferentiated diseases and common disease management.",
                focus_areas=["Health Check-up Interpretation", "Initial Diagnosis of Common Diseases", "Two-way Referral", "Chronic Disease Follow-up"],
                thinking_patterns=["Whole-person assessment", "Problem list prioritization", "Referral timing judgment"],
                metadata=ExpertMetadata(
                    specialty_tags=["General Practice", "GP", "Family Medicine"],
                    applicable_stages=["Initial", "Follow-up"],
                    risk_preference="Conservative",
                    target_populations=["Adult", "Elderly", "Child"],
                    system_focus=["Digestive", "Cardiovascular", "Respiratory", "Endocrine"],
                    confidence_domains=["Initial Diagnosis of Common Diseases", "Chronic Disease Management", "Health Consultation"]
                )
            ),
            ExpertUnit(
                id="gastro_surgery",
                name="Gastrointestinal Surgery Expert",
                specialty="Gastrointestinal Surgery",
                description="Specializes in surgical treatment of stomach, small intestine, colorectal, and anal diseases, especially tumors and acute abdomen.",
                focus_areas=["Gastrointestinal Tumors", "Appendicitis", "Intestinal Obstruction", "Hernia Repair"],
                thinking_patterns=["Acute abdomen localization", "TNM staging", "Surgical resection range assessment"],
                metadata=ExpertMetadata(
                    specialty_tags=["Gastrointestinal Surgery", "General Surgery", "Colorectal Surgery"],
                    applicable_stages=["Emergency", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Digestive System"],
                    confidence_domains=["Colorectal Cancer", "Acute Appendicitis", "Intestinal Obstruction"]
                )
            ),
            ExpertUnit(
                id="cardiothoracic_surgery",
                name="Cardiothoracic Surgery Expert",
                specialty="Cardiothoracic Surgery",
                description="Specializes in complex surgical treatment of intrathoracic organs.",
                focus_areas=["Lung Cancer Surgery", "Heart Valve Disease", "CABG", "Aortic Dissection"],
                thinking_patterns=["Preoperative cardiac function assessment", "Lung cancer staging and surgical plan", "Aortic lesion classification"],
                metadata=ExpertMetadata(
                    specialty_tags=["Cardiothoracic Surgery", "Chest Surgery", "Heart Surgery"],
                    applicable_stages=["Inpatient", "Emergency"],
                    risk_preference="Aggressive",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Cardiovascular", "Respiratory"],
                    confidence_domains=["Lung Cancer Resection", "Valve Replacement", "CABG"]
                )
            ),
            ExpertUnit(
                id="oncology",
                name="Oncology Expert",
                specialty="Oncology",
                description="Specializes in comprehensive medical treatment of various benign and malignant tumors, including chemotherapy, targeted therapy, and immunotherapy.",
                focus_areas=["Chemo/Radiotherapy Plans", "Cancer Screening", "Cancer Pain Management", "MDT"],
                thinking_patterns=["Tumor staging assessment", "Treatment plan selection", "Prognosis assessment"],
                metadata=ExpertMetadata(
                    specialty_tags=["Oncology", "Cancer"],
                    applicable_stages=["Follow-up", "Inpatient"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Digestive", "Respiratory", "Reproductive"],
                    confidence_domains=["Solid Tumor Chemotherapy", "Targeted Therapy", "ICI"]
                )
            ),
            ExpertUnit(
                id="cardiology",
                name="Cardiology Expert",
                specialty="Cardiology",
                description="Specializes in medical diagnosis and intervention of heart and vascular diseases, focusing on cardiovascular risk prevention.",
                focus_areas=["Hypertension", "Coronary Heart Disease", "Arrhythmia", "Heart Failure"],
                thinking_patterns=["Cardiovascular risk stratification", "ECG interpretation", "Intervention vs drug decision"],
                metadata=ExpertMetadata(
                    specialty_tags=["Cardiology", "Heart", "Cardiac"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient", "Emergency"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Cardiovascular System"],
                    confidence_domains=["ACS", "Heart Failure Management", "AF Anticoagulation"]
                )
            ),
            ExpertUnit(
                id="neurology",
                name="Neurology Expert",
                specialty="Neurology",
                description="Specializes in diagnosis and medical treatment of nervous system diseases.",
                focus_areas=["Cerebrovascular Disease", "Headache", "Epilepsy", "Parkinson's Disease"],
                thinking_patterns=["Neurological localization diagnosis", "Stroke time window assessment", "Cognitive function assessment"],
                metadata=ExpertMetadata(
                    specialty_tags=["Neurology", "Neuro"],
                    applicable_stages=["Initial", "Follow-up", "Inpatient", "Emergency"],
                    risk_preference="Neutral",
                    target_populations=["Adult", "Elderly"],
                    system_focus=["Nervous System"],
                    confidence_domains=["Ischemic Stroke", "Migraine", "Parkinson's Disease"]
                )
            ),
            ExpertUnit(
                id="nutrition",
                name="Nutrition Expert",
                specialty="Nutrition",
                description="Specializes in clinical nutrition assessment, perioperative nutrition support, and chronic disease dietary management.",
                focus_areas=["Malnutrition Assessment", "Enteral/Parenteral Nutrition", "Preoperative Nutrition Optimization", "Metabolic Support"],
                thinking_patterns=["Nutrition risk screening", "Energy requirement calculation", "Nutrition intervention plan formulation"],
                metadata=ExpertMetadata(
                    specialty_tags=["Nutrition", "Clinical Nutrition"],
                    applicable_stages=["Inpatient", "Follow-up"],
                    risk_preference="Conservative",
                    target_populations=["Adult", "Elderly", "Child"],
                    system_focus=["Digestive", "Endocrine"],
                    confidence_domains=["Malnutrition", "Perioperative Nutrition", "Enteral Nutrition"]
                )
            ),
        ]
        
        print("Initializing expert vector representations...")
        for expert in default_experts:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description} {' '.join(expert.focus_areas)}"
            expert.embedding = self._get_embedding(embedding_text)
            self.experts[expert.id] = expert
        
        print(f"Initialized {len(self.experts)} default expert units")
    
    def activate_experts(self, episode: DiagnosticEpisode, 
                        top_k: int = 5,
                        semantic_weight: float = 0.4,
                        tag_weight: float = 0.4,
                        stage_weight: float = 0.2) -> List[ExpertUnit]:
        """Episode-conditioned expert activation"""
        if episode.embedding is None:
            episode.embedding = self._get_embedding(episode.summary)
        
        scores = []
        for expert_id, expert in self.experts.items():
            # 1. Semantic similarity
            if expert.embedding:
                semantic_sim = self._cosine_similarity(episode.embedding, expert.embedding)
            else:
                semantic_sim = 0.0
            
            # 2. Tag matching
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
            
            # 3. Stage consistency
            stage_score = 1.0 if episode.stage in expert.metadata.applicable_stages else 0.3
            
            # Combined score
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
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        activated = []
        print(f"\nExpert Activation Ranking (top-{top_k}):")
        for i, (expert, score, details) in enumerate(scores[:top_k]):
            print(f"  {i+1}. {expert.name}: {score:.3f} "
                  f"(Semantic:{details['semantic']:.3f}, Tag:{details['tag']:.3f}, Stage:{details['stage']:.3f})")
            activated.append(expert)
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
        """Hybrid Activation Mode: Combines EEP semantic activation and Step-1 routing"""
        if episode.embedding is None:
            episode.embedding = self._get_embedding(episode.summary)
        
        # Specialty alias mapping (English)
        specialty_aliases = {
            "Cardiology": ["Heart", "Cardiac", "Cardiovascular"],
            "Gastroenterology": ["Gastro", "GI", "Digestive"],
            "Oncology": ["Cancer", "Tumor"],
            "Respiratory Medicine": ["Pulmonology", "Lung", "Respiratory"],
            "OB/GYN": ["Obstetrics", "Gynecology", "Maternity"],
            "Emergency Medicine": ["ER", "Emergency"],
            "Neurology": ["Neuro", "Brain"],
            "Hepatobiliary Surgery": ["Hepatobiliary", "Liver Surgery"],
            "Urology": ["Uro", "Renal Surgery"],
            "Orthopedics": ["Ortho", "Bone Surgery"],
            "Gastrointestinal Surgery": ["GI Surgery", "General Surgery", "Colorectal Surgery"],
            "Endocrinology": ["Endo", "Metabolism"],
            "Nephrology": ["Renal", "Kidney"],
            "General Practice": ["GP", "Family Medicine"],
        }
        
        # Normalize Step-1 recommendations
        normalized_step1 = set()
        for spec in step1_recommended:
            normalized_step1.add(spec)
            for key, aliases in specialty_aliases.items():
                if spec in aliases or key in spec:
                    normalized_step1.update(aliases)
                    normalized_step1.add(key)
        
        scores = []
        for expert_id, expert in self.experts.items():
            # EEP Score
            if expert.embedding:
                semantic_sim = self._cosine_similarity(episode.embedding, expert.embedding)
            else:
                semantic_sim = 0.0
            
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
            
            stage_score = 1.0 if episode.stage in expert.metadata.applicable_stages else 0.3
            
            eep_score = (
                semantic_weight * semantic_sim +
                tag_weight * tag_score +
                stage_weight * stage_score
            )
            
            # Step-1 Score
            step1_score = 0.0
            if step1_recommended:
                expert_specialty = expert.specialty
                expert_tags = expert.metadata.specialty_tags
                if expert_specialty in normalized_step1:
                    step1_score = 1.0
                else:
                    for tag in expert_tags:
                        if tag in normalized_step1:
                            step1_score = 1.0
                            break
                    if step1_score == 0:
                        for rec in step1_recommended:
                            if rec in expert_specialty or expert_specialty in rec:
                                step1_score = 0.8
                                break
                            for tag in expert_tags:
                                if rec in tag or tag in rec:
                                    step1_score = 0.6
                                    break
            
            if step1_recommended:
                total_score = (1 - step1_weight) * eep_score + step1_weight * step1_score
            else:
                total_score = eep_score
            
            scores.append((expert, total_score, {
                'eep': eep_score,
                'step1': step1_score,
                'semantic': semantic_sim,
                'tag': tag_score,
                'stage': stage_score
            }))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        activated = []
        print(f"\nHybrid Activation Ranking (top-{top_k}, Step-1 Weight={step1_weight:.0%}):")
        for i, (expert, score, details) in enumerate(scores[:top_k]):
            step1_mark = "✓" if details['step1'] > 0 else " "
            print(f"  {i+1}. [{step1_mark}] {expert.name}: {score:.3f} "
                  f"(EEP:{details['eep']:.3f}, Step1:{details['step1']:.1f})")
            activated.append(expert)
            expert.metadata.usage_count += 1
            expert.metadata.updated_at = datetime.now().isoformat()
        
        return activated
    
    def add_expert(self, expert: ExpertUnit, save: bool = True) -> bool:
        """Incrementally adds a new expert unit"""
        if expert.id in self.experts:
            print(f"Skip: Expert ID '{expert.id}' already exists")
            return False
        
        if expert.embedding is None:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description}"
            expert.embedding = self._get_embedding(embedding_text)
        
        max_sim = 0.0
        most_similar_expert = None
        for existing in self.experts.values():
            if existing.embedding:
                sim = self._cosine_similarity(expert.embedding, existing.embedding)
                if sim > max_sim:
                    max_sim = sim
                    most_similar_expert = existing
        
        if max_sim > 0.95 and most_similar_expert:
            print(f"Skip: New expert '{expert.name}' is highly similar to existing expert '{most_similar_expert.name}' (Sim={max_sim:.3f})")
            return False
        
        self.experts[expert.id] = expert
        print(f"Added new expert unit: {expert.name} (id={expert.id})")
        if save:
            self._save_pool()
        return True
    
    def update_expert(self, expert_id: str, updates: Dict[str, Any], save: bool = True) -> bool:
        """Updates an expert unit"""
        if expert_id not in self.experts:
            print(f"Expert {expert_id} does not exist")
            return False
        
        expert = self.experts[expert_id]
        for key, value in updates.items():
            if hasattr(expert, key):
                setattr(expert, key, value)
            elif hasattr(expert.metadata, key):
                setattr(expert.metadata, key, value)
        
        expert.metadata.updated_at = datetime.now().isoformat()
        if 'description' in updates or 'focus_areas' in updates:
            embedding_text = f"{expert.name} {expert.specialty} {expert.description}"
            expert.embedding = self._get_embedding(embedding_text)
        
        if save:
            self._save_pool()
        return True
    
    def evolve_from_divergence(self, 
                               divergence_info: Dict[str, Any],
                               episode: DiagnosticEpisode) -> Optional[ExpertUnit]:
        """Generates a new expert unit based on diagnostic divergence"""
        divergence_point = divergence_info.get('divergence_point', '')
        conflicting_views = divergence_info.get('conflicting_views', [])
        missing_perspective = divergence_info.get('missing_perspective', '')
        suggested_expert = divergence_info.get('suggested_expert', '')
        
        if not divergence_point:
            return None
        
        new_expert_prompt = f"""You are a medical expert system designer. Please design a new expert to fill the gaps in the current expert pool based on diagnostic divergence.

[Divergence Point]
{divergence_point}

[Missing Perspective]
{missing_perspective}

[Suggested New Expert Type]
{suggested_expert}

[Conflicting Views]
{json.dumps(conflicting_views, indent=2)}

[Case Background]
{episode.summary}

Please generate the expert configuration corresponding to the "Suggested New Expert Type".
Output in JSON format (all fields must be in English):

{{
    "id": "English ID, e.g., tumor_specialist or gastroenterology_functional",
    "name": "Expert Name, e.g., Oncology Specialist, Functional GI Expert",
    "specialty": "Specialty Name, e.g., Oncology, Functional Gastroenterology",
    "description": "Description, 50-100 words, explaining the expert's professional expertise",
    "focus_areas": ["Focus Area 1", "Focus Area 2", "Focus Area 3"],
    "thinking_patterns": ["Thinking Path 1", "Thinking Path 2"],
    "risk_preference": "Conservative/Neutral/Aggressive"
}}

Note:
1. 'name' and 'specialty' must correspond to the "Suggested New Expert Type".
2. All descriptions, areas, and paths must be in English.
3. Output JSON only, no explanation.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical expert system designer. You must output all expert configurations in English."},
                    {"role": "user", "content": new_expert_prompt}
                ],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]
            
            config = json.loads(result.strip())
            
            new_expert = ExpertUnit(
                id=config['id'],
                name=config['name'],
                specialty=config['specialty'],
                description=config['description'],
                focus_areas=config['focus_areas'],
                thinking_patterns=config.get('thinking_patterns', []),
                metadata=ExpertMetadata(
                    specialty_tags=[config['specialty']],
                    applicable_stages=["Initial", "Follow-up", "Inpatient"],
                    risk_preference=config.get('risk_preference', 'Neutral'),
                    target_populations=["Adult", "Elderly"],
                    system_focus=episode.tags.get('system_domain', []),
                    confidence_domains=config['focus_areas'][:3]
                )
            )
            
            if self.add_expert(new_expert):
                return new_expert
            
        except Exception as e:
            print(f"Failed to generate new expert: {e}")
        
        return None
    
    def get_expert_by_specialty(self, specialty: str) -> Optional[ExpertUnit]:
        """Gets expert by specialty name"""
        for expert in self.experts.values():
            if specialty in expert.metadata.specialty_tags or specialty == expert.specialty:
                return expert
        return None
    
    def get_all_experts(self) -> List[ExpertUnit]:
        """Gets all experts"""
        return list(self.experts.values())
    
    def get_specialty_mapping(self) -> Dict[str, ExpertUnit]:
        """Gets mapping from specialty name to expert"""
        mapping = {}
        for expert in self.experts.values():
            mapping[expert.specialty] = expert
            for tag in expert.metadata.specialty_tags:
                mapping[tag] = expert
        return mapping


def create_default_pool(pool_path: str = "expert_pool.json") -> EvolvingExpertPool:
    """Creates default expert resource pool"""
    return EvolvingExpertPool(pool_path)


if __name__ == "__main__":
    print("="*60)
    print("Evolving Expert Pool (EEP) Test")
    print("="*60)
    
    pool = EvolvingExpertPool()
    print(f"\nExpert pool contains {len(pool.experts)} expert units")
    
    from dataclasses import dataclass as dc
    @dc
    class MockPatient:
        patient_id: str = "test001"
        gender: str = "Female"
        age: int = 45
        chief_complaint: str = "Abdominal pain and distension for 3 days"
        history_of_present_illness: str = "Abdominal pain and distension appeared without obvious triggers 3 days ago, accompanied by nausea, no vomiting or fever"
        past_history: str = "History of hypertension for 5 years"
    
    patient = MockPatient()
    episode = DiagnosticEpisode.from_patient_info(patient)
    
    print(f"\nDiagnostic Episode:")
    print(f"  Keywords: {episode.keywords}")
    print(f"  System Domain: {episode.tags['system_domain']}")
    print(f"  Population: {episode.tags['population']}")
    
    print("\nActivating experts...")
    activated = pool.activate_experts(episode, top_k=5)
    print(f"\nActivated experts: {[e.name for e in activated]}")
