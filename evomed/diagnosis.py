"""
Hybrid Multi-Specialty Expert Agent Diagnosis System

Supports three expert activation modes:
1. Step-1 Route Mode: Matches experts based on LLM consultation suggestions
2. EEP Semantic Activation Mode: Activates experts based on Episode semantic similarity
3. Evolved Pool Mode: Uses the optimal expert pool evolved via Genetic Algorithms
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
import time

from evomed.evoexperts.prompt.step1_route import system_step1_prompt
from evomed.evoexperts.prompt.step2_ir import system_step2_prompt
from evomed.evoexperts.prompt.step3_diag import system_step3_prompt
from evomed.evoexperts.prompt.step4_agg import system_step4_prompt

# Import evolvable expert resource pool
from evomed.evoexperts.models.expert_pool import EvolvingExpertPool, DiagnosticEpisode, ExpertUnit

# Import knowledge retrieval services
from evomed.retrieval.knowledge import KnowledgeRetriever

API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

EXPERTS_CONFIG = [
    {
        "id": "obgyn",
        "name": "OB/GYN Expert",
        "specialty": "Obstetrics and Gynecology",
        "description": "Specializes in the diagnosis and treatment of female reproductive system diseases and perinatal care, focusing on women's health throughout their life cycle.",
        "focus_areas": ["Perinatal Management", "Gynecological Oncology", "Menstrual Disorders", "Reproductive Tract Inflammation"],
    },
    {
        "id": "gastroenterology",
        "name": "Gastroenterology Expert",
        "specialty": "Gastroenterology",
        "description": "Specializes in the medical diagnosis and treatment of diseases of the esophagus, stomach, intestines, liver, gallbladder, and pancreas, as well as endoscopic examinations.",
        "focus_areas": ["Gastritis and Ulcers", "Liver Diseases", "Functional Gastrointestinal Disorders", "Gastrointestinal Bleeding"],
    },
    {
        "id": "pediatrics",
        "name": "Pediatrics Expert",
        "specialty": "Pediatrics",
        "description": "Focuses on the growth, development, and disease diagnosis and treatment from newborn to adolescence, paying attention to the unique physiological and pathological characteristics of children.",
        "focus_areas": ["Respiratory Tract Infections", "Growth and Development Assessment", "Pediatric Digestive System", "Newborn Care"],
    },
    {
        "id": "endocrinology",
        "name": "Endocrinology Expert",
        "specialty": "Endocrinology",
        "description": "Specializes in the diagnosis and long-term management of hormone secretion abnormalities and metabolic diseases.",
        "focus_areas": ["Diabetes Management", "Thyroid Diseases", "Osteoporosis", "Obesity and Metabolic Syndrome"],
    },
    {
        "id": "hepatobiliary_surgery",
        "name": "Hepatobiliary Surgery Expert",
        "specialty": "Hepatobiliary Surgery",
        "description": "Specializes in the surgical treatment and perioperative management of liver, biliary tract, and pancreatic diseases.",
        "focus_areas": ["Cholelithiasis", "Liver Tumors", "Pancreatitis", "Biliary Obstruction"],
    },
    {
        "id": "orthopedics",
        "name": "Orthopedics Expert",
        "specialty": "Orthopedics",
        "description": "Specializes in the diagnosis, reduction, and surgical treatment of diseases of the motor system such as bones, joints, muscles, and ligaments.",
        "focus_areas": ["Fracture Trauma", "Arthritis", "Spine Diseases", "Sports Injuries"],
    },
    {
        "id": "respiratory",
        "name": "Respiratory Expert",
        "specialty": "Respiratory Medicine",
        "description": "Specializes in the medical diagnosis and treatment of respiratory system infections, airway diseases, and lung tumors.",
        "focus_areas": ["COPD", "Asthma Management", "Lung Nodules", "Pulmonary Infections"],
    },
    {
        "id": "emergency",
        "name": "Emergency Expert",
        "specialty": "Emergency Medicine",
        "description": "Specializes in the initial assessment, emergency resuscitation, and triage of acute illnesses, trauma, and various critical conditions.",
        "focus_areas": ["Vital Signs Maintenance", "Acute Poisoning", "Multiple Trauma", "CPR"],
    },
    {
        "id": "urology",
        "name": "Urology Expert",
        "specialty": "Urology",
        "description": "Specializes in the surgical and minimally invasive treatment of the urinary system (kidneys, bladder, urinary tract) and male reproductive system diseases.",
        "focus_areas": ["Urinary Calculi", "Prostate Diseases", "Urinary Tract Tumors", "Urinary Tract Infections"],
    },
    {
        "id": "general_practice",
        "name": "General Practice Expert",
        "specialty": "General Practice",
        "description": "Provides comprehensive and continuous basic medical services, specializing in the initial diagnosis of undifferentiated diseases and the management of common illnesses.",
        "focus_areas": ["Health Check-up Interpretation", "Initial Diagnosis of Common Diseases", "Two-way Referral", "Long-term Follow-up of Chronic Diseases"],
    },
    {
        "id": "gastro_surgery",
        "name": "Gastrointestinal Surgery Expert",
        "specialty": "Gastrointestinal Surgery",
        "description": "Specializes in the surgical treatment of stomach, small intestine, colorectal, and anal diseases, especially in the handling of tumors and acute abdomen.",
        "focus_areas": ["Gastrointestinal Tumors", "Appendicitis", "Intestinal Obstruction", "Hernia Repair"],
    },
    {
        "id": "cardiothoracic_surgery",
        "name": "Cardiothoracic Surgery Expert",
        "specialty": "Cardiothoracic Surgery",
        "description": "Specializes in complex surgical treatments for intrathoracic organs (heart, great vessels, lungs, esophagus).",
        "focus_areas": ["Lung Cancer Surgery", "Heart Valve Disease", "CABG", "Aortic Dissection"],
    },
    {
        "id": "oncology",
        "name": "Oncology Expert",
        "specialty": "Oncology",
        "description": "Specializes in the comprehensive medical treatment of various benign and malignant tumors, including chemotherapy, targeted therapy, and immunotherapy.",
        "focus_areas": ["Chemo/Radiotherapy Plans", "Cancer Screening", "Cancer Pain Management", "MDT"],
    },
    {
        "id": "cardiology",
        "name": "Cardiology Expert",
        "specialty": "Cardiology",
        "description": "Specializes in the medical diagnosis, treatment, and interventional therapy of heart and vascular diseases, focusing on cardiovascular risk prevention and control.",
        "focus_areas": ["Hypertension", "Coronary Heart Disease", "Arrhythmia", "Heart Failure"],
    },
    {
        "id": "rheumatology",
        "name": "Rheumatology Expert",
        "specialty": "Rheumatology",
        "description": "Specializes in the diagnosis and long-term management of various rheumatic diseases and autoimmune diseases.",
        "focus_areas": ["Rheumatoid Arthritis", "SLE", "Ankylosing Spondylitis", "Gout"],
    },
    {
        "id": "neurology",
        "name": "Neurology Expert",
        "specialty": "Neurology",
        "description": "Specializes in the medical diagnosis and treatment of central nervous system, peripheral nervous system, and skeletal muscle diseases.",
        "focus_areas": ["Cerebrovascular Disease", "Epilepsy", "Parkinson's Disease", "Peripheral Neuropathy"],
    }
]


@dataclass
class PatientInfo:
    """Patient Information Data Class"""
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
        """Formats patient info into a prompt string"""
        return f"""
Patient ID: {self.patient_id}
Gender: {self.gender}
Age: {self.age} years old
Admitting Department: {self.department}

[Chief Complaint]
{self.chief_complaint}

[History of Present Illness]
{self.history_of_present_illness}

[Past Medical History]
{self.past_history}

[Personal History]
{self.personal_history}

[Physical Examination]
{self.physical_examination}

[Laboratory Results]
{self.labs}

[Imaging Results]
{self.imaging}
"""


class DiagnosticPipeline:
    """
    Multi-Disciplinary Diagnostic Pipeline
    
    Supports three expert activation modes:
    - 'step1_route': Specialty matching based on Step-1 consultation suggestions
    - 'eep_semantic': Activation based on EEP semantic similarity (recommended)
    - 'evolved_pool': Uses the optimal expert pool evolved via genetic algorithms
    
    Supports three major knowledge sources:
    - RAG Medical Guideline Retrieval
    - Experience Library Retrieval (A-Mem)
    - Case Library Retrieval (Engine)
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
        memory_db_root: str = "exp/repository/memory_db",
        experience_collection: str = "experience_100000",
        case_collection: str = "case_100000",
    ):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
        self.activation_mode = activation_mode
        self.resource = """[Medical Resources]
Available Departments: Internal Medicine, Surgery, Emergency, Cardiology, Gastroenterology, Respiratory, Neurology, Oncology, Orthopedics, Urology, Hepatobiliary Surgery, OB/GYN, Pediatrics, etc.
Knowledge Bases: Medical Guidelines, Clinical Experience Library, Similar Case Library.
Note: This system provides diagnostic assistance only and does not replace professional medical activities."""
        
        # Initialize Evolvable Expert Pool (EEP)
        if activation_mode == "eep_semantic":
            print("Initializing Evolvable Expert Pool (EEP)...")
            self.expert_pool = EvolvingExpertPool(expert_pool_path)
            self.experts = EXPERTS_CONFIG  # Keep for compatibility
        else:
            self.expert_pool = None
            self.experts = EXPERTS_CONFIG
        
        # Initialize GA Evolved Pool
        self.evolved_pool = None
        self.evolved_pool_path = evolved_pool_path
        if activation_mode == "evolved_pool":
            print("Loading GA evolved expert pool...")
            self.evolved_pool = self._load_evolved_pool(evolved_pool_path)
            if self.evolved_pool:
                print(f"  ✅ Successfully loaded {len(self.evolved_pool)} evolved expert prompts")
            else:
                print("  ⚠️ Evolved expert pool not found, falling back to default mode")
                self.activation_mode = "step1_route"
        
        # Initialize Knowledge Retrieval Services
        print("Initializing knowledge retrieval services...")
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
        
        # Build specialty name to expert config mapping
        self.specialty_mapping = self._build_specialty_mapping()
    
    def _load_evolved_pool(self, path: str) -> Optional[List[Dict]]:
        """Loads the expert pool evolved via genetic algorithms"""
        if not os.path.exists(path):
            print(f"  Warning: Evolved pool file does not exist: {path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            
            # Validate format
            if not isinstance(pool, list) or len(pool) == 0:
                print(f"  Warning: Evolved pool format is invalid")
                return None
            
            # Ensure each expert has required fields
            for i, expert in enumerate(pool):
                if 'prompt' not in expert:
                    print(f"  Warning: Expert {i} is missing the 'prompt' field")
                    return None
                # Add ID if missing
                if 'id' not in expert:
                    expert['id'] = f"evolved_expert_{i}"
                # Add name if missing
                if 'name' not in expert:
                    expert['name'] = f"Evolved Expert-{i+1}"
                    
            return pool
            
        except Exception as e:
            print(f"  Warning: Failed to load evolved pool: {e}")
            return None
    
    def _build_specialty_mapping(self) -> Dict[str, Dict]:
        """Builds a specialty name mapping table, supporting various aliases"""
        mapping = {}
        
        # Alias mapping (English)
        aliases = {
            "Obstetrics and Gynecology": ["OB/GYN", "Gynecology", "Obstetrics", "Maternity"],
            "Gastroenterology": ["Gastro", "GI", "Digestive"],
            "Pediatrics": ["Peds", "Child Health"],
            "Endocrinology": ["Endo", "Metabolism"],
            "Hepatobiliary Surgery": ["Hepatobiliary", "Liver Surgery"],
            "Orthopedics": ["Ortho", "Bone Surgery", "Traumatology"],
            "Respiratory Medicine": ["Pulmonology", "Lung", "Respiratory"],
            "Emergency Medicine": ["ER", "Emergency", "Critical Care"],
            "Urology": ["Uro", "Renal Surgery"],
            "General Practice": ["GP", "Family Medicine", "Internal Medicine"],
            "Gastrointestinal Surgery": ["GI Surgery", "General Surgery", "Colorectal Surgery"],
            "Cardiothoracic Surgery": ["Chest Surgery", "Heart Surgery", "Thoracic Surgery"],
            "Oncology": ["Cancer", "Tumor"],
            "Cardiology": ["Heart", "Cardiac"],
            "Rheumatology": ["Rheuma", "Autoimmune"],
            "Neurology": ["Neuro", "Brain"],
        }
        
        for expert in self.experts:
            specialty = expert['specialty']
            mapping[specialty] = expert
            if specialty in aliases:
                for alias in aliases[specialty]:
                    mapping[alias] = expert
        
        return mapping
    
    def _extract_recommended_specialties(self, step1_output: str) -> List[str]:
        """Extracts recommended specialties from Step-1 output"""
        extract_prompt = f"""From the following consultation/referral suggestions, extract all recommended specialty/department names.

[Step-1 Output]
{step1_output}

[Available Specialties]
{', '.join([e['specialty'] for e in self.experts])}

Please output the list of specialty names in order of priority.
Output in JSON array format ONLY, without any explanation. Example: ["Gastrointestinal Surgery", "Cardiology", "Respiratory Medicine"]
"""
        
        result = self._call_llm(extract_prompt)
        
        try:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\
", 1)[1]
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
        """Fallback: Extract specialties via keyword matching"""
        found = []
        for specialty in self.specialty_mapping.keys():
            if specialty in text and specialty not in found:
                expert = self.specialty_mapping[specialty]
                if expert['specialty'] not in found:
                    found.append(expert['specialty'])
        return found
    
    def _match_experts_by_specialties(self, specialties: List[str]) -> List[Dict]:
        """Matches expert configurations based on the specialty name list"""
        matched_experts = []
        matched_ids = set()
        
        for specialty in specialties:
            # Direct match
            if specialty in self.specialty_mapping:
                expert = self.specialty_mapping[specialty]
                if expert['id'] not in matched_ids:
                    matched_experts.append(expert)
                    matched_ids.add(expert['id'])
            else:
                # Fuzzy match
                for key, expert in self.specialty_mapping.items():
                    if key in specialty or specialty in key:
                        if expert['id'] not in matched_ids:
                            matched_experts.append(expert)
                            matched_ids.add(expert['id'])
                            break
        
        return matched_experts
    
    def _call_llm(self, system_prompt: str, max_retries: int = 3) -> str:
        """Calls the LLM API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a professional medical diagnostic assistant agent."},
                        {"role": "user", "content": system_prompt}
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
    
    def step1_route(self, patient: PatientInfo) -> Dict[str, Any]:
        """
        Step-1: Consultation/Referral Planning
        Input: Medical Resources r, Patient Info p
        Output: Priority list of specialties, urgency level, high-risk exclusion list
        """
        print("\
" + "="*60)
        print("[Step-1] Consultation/Referral Planning")
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
        Step-2: Expert Semantic Rewriting
        Input: Medical Resources r, Patient Info p, Expert Config e
        Output: Medical-style rewritten segment, structured retrieval element summary
        """
        expert_info = f"""
Expert Role: {expert['name']}
Specialty: {expert['specialty']}
Description: {expert['description']}
Focus Areas: {', '.join(expert['focus_areas'])}
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
        Step-3: Expert Differential Diagnosis
        Input: Medical Resources r, Patient Info p, Expert Config e, Reference ref
        Output: List of differential diagnoses, risks and warning signals, next examination directions
        """
        expert_info = f"""
Expert Role: {expert['name']}
Specialty: {expert['specialty']}
Description: {expert['description']}
Focus Areas: {', '.join(expert['focus_areas'])}

[Rewritten Patient Info from Expert's Perspective]
{rewritten_info}
"""
        
        # Reference retrieval
        if not reference and auto_retrieve:
            reference = self.knowledge_retriever.retrieve_for_expert(
                rewritten_query=rewritten_info,
                expert_specialty=expert['specialty'],
                rag_k=3,
                experience_k=3,
                case_k=3
            )
        elif not reference:
            reference = "[No additional reference material available]"
        
        # Use custom or default template
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
            "reference_used": reference
        }
    
    def step4_aggregate(self, patient: PatientInfo, 
                        expert_opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step-4: Multi-Expert Consensus Aggregation
        Input: Medical Resources r, Patient Info p, Collection of Expert Opinions
        Output: Comprehensive diagnostic conclusion, explanation of disagreements, comprehensive risk assessment, next action suggestions
        """
        print("\
" + "="*60)
        print("[Step-4] Multi-Expert Consensus Aggregation")
        print("="*60)
        
        # Aggregate all expert opinions
        experts_summary = ""
        for opinion in expert_opinions:
            experts_summary += f"""
{'='*40}
Diagnostic Opinion from [{opinion['expert_name']}]
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
        """Converts ExpertUnit to a compatible dictionary format"""
        return {
            'id': expert_unit.id,
            'name': expert_unit.name,
            'specialty': expert_unit.specialty,
            'description': expert_unit.description,
            'focus_areas': expert_unit.focus_areas,
            'thinking_patterns': getattr(expert_unit, 'thinking_patterns', []),
            'risk_preference': expert_unit.metadata.risk_preference if expert_unit.metadata else 'Neutral'
        }
    
    def _activate_experts_eep(
        self, 
        patient: PatientInfo, 
        top_k: int = 5,
        step1_output: str = "",
        step1_weight: float = 0.3
    ) -> Tuple[List[Dict], DiagnosticEpisode]:
        """
        Selects experts using EEP hybrid activation mode
        
        Combines two signals:
        1. EEP Semantic Activation (1 - step1_weight): Based on Episode semantic similarity
        2. Step-1 Route Recommendation (step1_weight): Based on LLM consultation suggestions
        """
        print("\
" + "-"*60)
        print(f"[EEP Hybrid Activation] Semantic Activation + Step-1 Routing (Weight: {step1_weight:.0%})")
        print("-"*60)
        
        # Create Diagnostic Episode
        episode = DiagnosticEpisode.from_patient_info(patient)
        
        print(f"Episode ID: {episode.episode_id}")
        print(f"Keywords: {episode.keywords}")
        print(f"System Domain: {episode.tags.get('system_domain', [])}")
        print(f"Population: {episode.tags.get('population', [])}")
        
        # Extract recommended specialties from Step-1
        step1_recommended = []
        if step1_output:
            step1_recommended = self._extract_recommended_specialties(step1_output)
            print(f"Step-1 Recommended Specialties: {step1_recommended}")
        
        # Activate experts via EEP
        activated_units = self.expert_pool.activate_experts_hybrid(
            episode, 
            step1_recommended=step1_recommended,
            step1_weight=step1_weight,
            top_k=top_k
        )
        
        # Convert to compatible format
        selected_experts = [self._expert_unit_to_dict(unit) for unit in activated_units]
        
        return selected_experts, episode
    
    def _activate_experts_step1(self, step1_output: str) -> List[Dict]:
        """
        Selects experts using pure Step-1 routing mode
        """
        print("\
" + "-"*60)
        print("[Step-1 Routing] Matching experts based on consultation suggestions...")
        print("-"*60)
        
        recommended_specialties = self._extract_recommended_specialties(step1_output)
        selected_experts = self._match_experts_by_specialties(recommended_specialties)
        
        print(f"Step-1 Recommended Specialties: {recommended_specialties}")
        print(f"Matched Experts: {[e['name'] for e in selected_experts]}")
        
        if not selected_experts:
            print("Warning: No matching experts found, using General Practice Expert")
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
        Activates experts using the evolved expert pool
        """
        print("\
" + "-"*60)
        print(f"[Evolved Pool Activation] Selection Strategy: {selection_strategy}")
        print("-"*60)
        
        if not self.evolved_pool:
            print("Warning: Evolved pool not loaded, falling back to Step-1 routing")
            base_experts = self._activate_experts_step1(step1_output)
            return base_experts, [None] * len(base_experts)
        
        recommended_specialties = self._extract_recommended_specialties(step1_output)
        base_experts = self._match_experts_by_specialties(recommended_specialties)
        
        if not base_experts:
            for expert in self.experts:
                if expert['id'] == 'general_practice':
                    base_experts = [expert]
                    break
        
        base_experts = base_experts[:top_k]
        evolved_prompts = []
        
        if selection_strategy == "hybrid":
            for expert in base_experts:
                candidates = []
                for evolved in self.evolved_pool:
                    if (evolved.get('specialty') == expert['specialty'] or 
                        expert['id'] in evolved.get('id', '')):
                        candidates.append(evolved)
                
                if candidates:
                    best_candidate = max(candidates, key=lambda x: x.get('fitness', 0))
                    evolved_prompts.append(best_candidate)
                else:
                    print(f"  ⚠️ No evolved expert found for specialty '{expert['specialty']}', using default prompt")
                    evolved_prompts.append(None)
                    
        elif selection_strategy == "top_fitness":
            sorted_pool = sorted(self.evolved_pool, key=lambda x: x.get('fitness', 0), reverse=True)
            for i, expert in enumerate(base_experts):
                if i < len(sorted_pool):
                    evolved_prompts.append(sorted_pool[i])
                else:
                    evolved_prompts.append(sorted_pool[0] if sorted_pool else None)
                    
        else: # diversity or others
            for i, expert in enumerate(base_experts):
                idx = i % len(self.evolved_pool)
                evolved_prompts.append(self.evolved_pool[idx])
        
        print(f"Base Specialty Experts: {[e['name'] for e in base_experts]}")
        print(f"Evolved Prompt Assignment:")
        for i, (expert, prompt_config) in enumerate(zip(base_experts, evolved_prompts)):
            if prompt_config:
                fitness = prompt_config.get('fitness', 'N/A')
                prompt_id = prompt_config.get('id', f'prompt_{i}')
                match_status = "✅" if prompt_config.get('specialty') == expert['specialty'] else "⚠️(Mismatch)"
                print(f"  [{i+1}] {expert['name']} <- {prompt_id} {match_status} (Fitness: {fitness:.4f if isinstance(fitness, float) else fitness})")
            else:
                print(f"  [{i+1}] {expert['name']} <- Default Prompt")
        
        return base_experts, evolved_prompts
    
    def run_pipeline(self, patient: PatientInfo, top_k: int = 8, 
                     evolved_selection_strategy: str = "hybrid") -> Dict[str, Any]:
        """Runs the complete four-step pipeline"""
        results = {
            "patient_id": patient.patient_id,
            "activation_mode": self.activation_mode,
            "steps": {}
        }
        
        # Step-1
        step1_result = self.step1_route(patient)
        results["steps"]["step1"] = step1_result
        
        episode = None
        evolved_prompts = None
        
        if self.activation_mode == "eep_semantic" and self.expert_pool:
            selected_experts, episode = self._activate_experts_eep(
                patient, 
                top_k=top_k,
                step1_output=step1_result["output"],
                step1_weight=0.3
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
            selected_experts = self._activate_experts_step1(step1_result["output"])
            results["steps"]["routing"] = {
                "mode": "step1_route",
                "selected_experts": [e['name'] for e in selected_experts]
            }
        
        print(f"\
Activated Experts ({len(selected_experts)}): {[e['name'] for e in selected_experts]}")
        
        # Step-2 & Step-3
        expert_opinions = []
        
        for idx, expert in enumerate(selected_experts):
            print("\
" + "-"*60)
            print(f"[Step-2 & Step-3] [{idx+1}/{len(selected_experts)}] {expert['name']} Analyzing...")
            print("-"*60)
            
            step2_result = self.step2_semantic_rewrite(patient, expert)
            print(f"\
[{expert['name']}] Semantic rewriting complete")
            
            custom_prompt = None
            evolved_prompt_id = None
            if evolved_prompts and idx < len(evolved_prompts) and evolved_prompts[idx]:
                custom_prompt = evolved_prompts[idx].get('prompt')
                evolved_prompt_id = evolved_prompts[idx].get('id')
                print(f"  📝 Using evolved prompt: {evolved_prompt_id}")
            
            step3_result = self.step3_diagnosis(
                patient, expert, 
                step2_result["output"],
                custom_prompt_template=custom_prompt
            )
            print(f"\
[{expert['name']}] Differential Diagnosis:")
            print(step3_result["output"])
            
            opinion_record = {
                "expert_id": expert['id'],
                "expert_name": expert['name'],
                "specialty": expert['specialty'],
                "rewrite": step2_result["output"],
                "diagnosis": step3_result["output"],
                "output": step3_result["output"]
            }
            
            if evolved_prompt_id:
                opinion_record["evolved_prompt_id"] = evolved_prompt_id
            
            expert_opinions.append(opinion_record)
        
        results["steps"]["expert_opinions"] = expert_opinions
        
        # Step-4
        step4_result = self.step4_aggregate(patient, expert_opinions)
        results["steps"]["step4"] = step4_result
        
        # Step-5 (Optional): Divergence detection and expert evolution
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
        
        # Save pool
        if self.activation_mode == "eep_semantic" and self.expert_pool:
            self.expert_pool._save_pool()
        
        return results
    
    def _detect_and_evolve_with_reanalysis(self, patient: PatientInfo, step4_output: str, 
                                            episode: DiagnosticEpisode, 
                                            expert_opinions: List[Dict],
                                            results: Dict) -> Optional[Dict]:
        """Detects diagnostic divergence, generates new experts, and performs re-analysis"""
        print("\
" + "-"*60)
        print("[Step-5] Divergence Detection and Expert Evolution")
        print("-"*60)
        
        detect_prompt = f"""Analyze the following multi-expert consensus aggregation result and determine if a new expert perspective should be introduced.

[Step-4 Aggregation Result]
{step4_output}

[Participating Experts]
{', '.join([e['expert_name'] for e in expert_opinions])}

Please analyze:
1. Is there significant disagreement among experts?
2. Is there an important perspective currently missing?
3. Is a new expert unit needed to fill the gap?

Output in JSON format:
{{
    "has_significant_divergence": true/false,
    "divergence_point": "Description of divergence (empty if none)",
    "conflicting_views": ["View 1", "View 2"],
    "missing_perspective": "Description of missing perspective (empty if none)",
    "suggested_new_expert": "Suggested type of new expert (English, empty if none)",
    "evolution_priority": "high/medium/low/none"
}}

Output JSON only."""
        
        try:
            result = self._call_llm(detect_prompt)
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\
", 1)[1]
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0]
            
            analysis = json.loads(result.strip())
            
            print(f"Divergence detection results:")
            print(f"  - Significant divergence: {analysis.get('has_significant_divergence', False)}")
            print(f"  - Divergence point: {analysis.get('divergence_point', 'None')}")
            print(f"  - Evolution priority: {analysis.get('evolution_priority', 'none')}")
            
            if (analysis.get('has_significant_divergence') and 
                analysis.get('evolution_priority') in ['high', 'medium']):
                
                print(f"\
Significant divergence detected, attempting to generate new expert...")
                print(f"  - Missing perspective: {analysis.get('missing_perspective', 'Unknown')}")
                print(f"  - Suggested expert: {analysis.get('suggested_new_expert', 'Unknown')}")
                
                divergence_info = {
                    'divergence_point': analysis.get('divergence_point', ''),
                    'conflicting_views': analysis.get('conflicting_views', []),
                    'missing_perspective': analysis.get('missing_perspective', ''),
                    'suggested_expert': analysis.get('suggested_new_expert', '')
                }
                
                new_expert = self.expert_pool.evolve_from_divergence(divergence_info, episode)
                
                if new_expert:
                    print(f"\
✅ Successfully generated new expert unit: {new_expert.name}")
                    print(f"   Specialty: {new_expert.specialty}")
                    print(f"   Focus Areas: {', '.join(new_expert.focus_areas)}")
                    
                    print("\
" + "="*60)
                    print(f"[New Expert Participation] {new_expert.name} joining consultation")
                    print("="*60)
                    
                    new_expert_dict = self._expert_unit_to_dict(new_expert)
                    
                    print(f"\
[{new_expert.name}] Performing semantic rewriting...")
                    step2_new = self.step2_semantic_rewrite(patient, new_expert_dict)
                    
                    print(f"\
[{new_expert.name}] Performing differential diagnosis...")
                    step3_new = self.step3_diagnosis(patient, new_expert_dict, step2_new["output"])
                    
                    new_opinion = {
                        "expert_id": new_expert.id,
                        "expert_name": new_expert.name,
                        "specialty": new_expert.specialty,
                        "rewrite": step2_new["output"],
                        "diagnosis": step3_new["output"],
                        "output": step3_new["output"],
                        "is_evolved": True
                    }
                    
                    all_opinions = expert_opinions + [new_opinion]
                    results["steps"]["expert_opinions"] = all_opinions
                    
                    print("\
" + "="*60)
                    print("[Step-4 Re-aggregation] Aggregation with new expert opinion")
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
                    print(f"\
⚠️ Failed to generate new expert (may duplicate existing expert)")
                    return {
                        "divergence_detected": True,
                        "analysis": analysis,
                        "new_expert_created": False,
                        "reanalysis_performed": False
                    }
            else:
                print("No expert pool evolution needed at this time.")
                return {
                    "divergence_detected": False,
                    "analysis": analysis,
                    "reanalysis_performed": False
                }
                
        except Exception as e:
            print(f"Divergence detection failed: {e}")
            return None


def parse_patient_from_row(row: pd.Series) -> PatientInfo:
    """Parses patient information from a DataFrame row"""
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
    
    chief_complaint = history.get('Chief Complaint', '')
    hpi = history.get('History of Present Illness', '')
    past_history = history.get('Past Medical History', '')
    personal_history = history.get('Personal History', '')
    
    physical_exam = row.get('physical_examination_compressed', 
                           row.get('physical_examination', ''))
    if pd.isna(physical_exam):
        physical_exam = ''
    
    labs = row.get('labs_compressed', row.get('labs_lite', row.get('labs', '')))
    if pd.isna(labs):
        labs = ''
    
    imaging = row.get('exam_lite', row.get('exam', ''))
    if pd.isna(imaging):
        imaging = ''
    
    gender = row.get('gender', '')
    if gender == 'M':
        gender = 'Male'
    elif gender == 'F':
        gender = 'Female'
    
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
    """Main Function"""
    print("="*70)
    print("Hybrid Multi-Specialty Expert Agent Diagnosis System")
    print(f"Mode: {activation_mode}")
    if activation_mode == "evolved_pool":
        print(f"Evolved Pool: {evolved_pool_path}")
        print(f"Selection Strategy: {evolved_selection_strategy}")
    print("="*70)
    
    # NOTE: The data file 'guilin_inpatient_extracted_10000.xlsx' was removed as part of cleanup.
    # This script is now intended to be used with other data sources or via the API/Web UI.
    try:
        df = pd.read_excel('guilin_inpatient_extracted_10000.xlsx')
        print(f"Loaded {len(df)} patient records")
    except FileNotFoundError:
        print("Data file 'guilin_inpatient_extracted_10000.xlsx' not found. Inference mode will use provided patient data.")
        df = pd.DataFrame()
    
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic" if activation_mode == "train_ga" else activation_mode,
        evolved_pool_path=evolved_pool_path,
        enable_rag=enable_rag,
        enable_experience=enable_experience,
        enable_case=enable_case,
    )
    
    if activation_mode == "train_ga":
        if df.empty:
            print("Cannot run GA training without data file.")
            return
        
        from evomed.evoexperts.optimizer import GeneticPromptOptimizer
        
        valid_mask = df['is_history_cleaned'] == True
        valid_df = df[valid_mask]
        if len(valid_df) == 0:
            valid_df = df.iloc[:10]
            
        train_patients = []
        print("Parsing training data...")
        for i in range(min(50, len(valid_df))):
            try:
                p = parse_patient_from_row(valid_df.iloc[i])
                train_patients.append(p)
            except Exception:
                pass
        
        optimizer = GeneticPromptOptimizer(
            base_prompt=system_step3_prompt,
            pipeline=pipeline,
            population_size=32,
            elitism_count=6
        )
        
        best_expert_pool = optimizer.run_evolution(train_patients, generations=generations, top_k_return=32)
        pool_file = "outputs/moa_optimized_expert_pool_64.json"
        with open(pool_file, 'w', encoding='utf-8') as f:
            json.dump(best_expert_pool, f, ensure_ascii=False, indent=2)
            
        print(f"\
🏆 Evolution complete! Optimal expert pool saved to {pool_file}")
        return best_expert_pool

    # Inference mode requires patient data. If df is empty, this part will fail or needs manual patient input.
    if df.empty:
        print("Inference mode: Please use the Web UI or provide patient data via API.")
        return

    idx = min(patient_index, len(df) - 1)
    sample_row = df.iloc[idx]
    patient = parse_patient_from_row(sample_row)
    
    results = pipeline.run_pipeline(
        patient, 
        top_k=top_k,
        evolved_selection_strategy=evolved_selection_strategy
    )
    
    output_file = f"diagnosis_result_{patient.patient_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\
Diagnosis complete! Result saved to: {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Multi-Specialty Expert Agent Diagnosis System")
    parser.add_argument("--mode", type=str, default="eep_semantic",
                       choices=["eep_semantic", "step1_route", "evolved_pool", "train_ga"],
                       help="Run mode")
    parser.add_argument("--top_k", type=int, default=8, help="Number of activated experts")
    parser.add_argument("--patient", type=int, default=0, help="Patient index")
    parser.add_argument("--generations", type=int, default=10, help="Evolution generations")
    parser.add_argument("--evolved_pool", type=str, default="outputs/moa_optimized_expert_pool_64.json")
    parser.add_argument("--evolved_strategy", type=str, default="hybrid")
    parser.add_argument("--no_rag", action="store_true")
    parser.add_argument("--no_experience", action="store_true")
    parser.add_argument("--no_case", action="store_true")
    
    args = parser.parse_args()
    
    main(
        activation_mode=args.mode, 
        top_k=args.top_k, 
        patient_index=args.patient,
        enable_rag=not args.no_rag,
        enable_experience=not args.no_experience,
        enable_case=not args.no_case,
        generations=args.generations,
        evolved_pool_path=args.evolved_pool,
        evolved_selection_strategy=args.evolved_strategy
    )
