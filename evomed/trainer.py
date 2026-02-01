"""
Specialty Track Genetic Algorithm Evolution Script (MOA Mode)

Trains expert pools for 16 departments respectively. For each department:
1. Select 5 cases from the dataset for that department as the validation set.
2. Perform 10 generations of GA evolution based on the expert template for that department (evaluated in the MOA multi-disciplinary consultation process).
3. Extract the Top 4 optimal experts (including complete expert settings).

Finally merged into a 64-expert pool.
"""

import json
import pandas as pd
import os
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Import main process modules
from evomed.diagnosis import (
    DiagnosticPipeline,
    PatientInfo,
    parse_patient_from_row
)
from evomed.evoexperts.optimizer import GeneticPromptOptimizer
from evomed.evoexperts.prompt.step3_diag import system_step3_prompt


# Complete expert definition templates for 16 departments
EXPERT_TEMPLATES = {
    "Obstetrics and Gynecology": {
        "id": "obgyn",
        "name": "OB/GYN Expert",
        "specialty": "Obstetrics and Gynecology",
        "description": "Specializes in the diagnosis and treatment of female reproductive system diseases and perinatal care, focusing on women's health throughout their life cycle.",
        "focus_areas": ["Pregnancy Management", "Gynecological Oncology", "Menstrual Disorders", "Reproductive Tract Infections"],
        "thinking_patterns": ["Menstrual Cycle Assessment", "Pregnancy-related Differentiation", "Hormone Level Analysis", "Pelvic Imaging Interpretation"],
    },
    "Gastroenterology": {
        "id": "gastroenterology",
        "name": "Gastroenterology Expert",
        "specialty": "Gastroenterology",
        "description": "Specializes in the medical diagnosis and treatment of diseases of the esophagus, stomach, intestines, liver, gallbladder, and pancreas, as well as endoscopic examinations.",
        "focus_areas": ["Gastritis and Ulcers", "Liver Diseases", "Functional Gastrointestinal Disorders", "Gastrointestinal Bleeding"],
        "thinking_patterns": ["Abdominal Pain Localization Analysis", "Liver Function Assessment", "Endoscopic Indication Judgment", "Helicobacter Pylori Screening"],
    },
    "Pediatrics": {
        "id": "pediatrics",
        "name": "Pediatrics Expert",
        "specialty": "Pediatrics",
        "description": "Focuses on growth, development, and disease diagnosis and treatment from newborn to adolescence, paying attention to the unique physiological and pathological characteristics of children.",
        "focus_areas": ["Respiratory Infections", "Growth Assessment", "Pediatric Digestive System", "Newborn Care"],
        "thinking_patterns": ["Age-related Differentiation", "Growth Curve Assessment", "Vaccination History", "Feeding Method Analysis"],
    },
    "Endocrinology": {
        "id": "endocrinology",
        "name": "Endocrinology Expert",
        "specialty": "Endocrinology",
        "description": "Specializes in the diagnosis and long-term management of hormone secretion abnormalities and metabolic diseases.",
        "focus_areas": ["Diabetes Management", "Thyroid Diseases", "Osteoporosis", "Obesity and Metabolic Syndrome"],
        "thinking_patterns": ["Blood Glucose Profile Analysis", "Thyroid Function Interpretation", "Insulin Function Assessment", "Metabolic Indicator Integration"],
    },
    "Hepatobiliary Surgery": {
        "id": "hepatobiliary_surgery",
        "name": "Hepatobiliary Surgery Expert",
        "specialty": "Hepatobiliary Surgery",
        "description": "Specializes in the surgical treatment and perioperative management of liver, biliary tract, and pancreatic diseases.",
        "focus_areas": ["Cholelithiasis", "Liver Tumors", "Pancreatitis", "Biliary Obstruction"],
        "thinking_patterns": ["Jaundice Differentiation", "Liver Function Reserve Assessment", "Surgical Indication Judgment", "Imaging Staging"],
    },
    "Orthopedics": {
        "id": "orthopedics",
        "name": "Orthopedics Expert",
        "specialty": "Orthopedics",
        "description": "Specializes in the diagnosis, reduction, and surgical treatment of diseases of the motor system such as bones, joints, muscles, and ligaments.",
        "focus_areas": ["Fracture Trauma", "Arthritis", "Spinal Diseases", "Sports Injuries"],
        "thinking_patterns": ["Injury Mechanism Analysis", "X-ray Fracture Classification", "Joint Range of Motion Assessment", "Nerve Injury Screening"],
    },
    "Respiratory Medicine": {
        "id": "respiratory",
        "name": "Respiratory Expert",
        "specialty": "Respiratory Medicine",
        "description": "Specializes in the medical diagnosis and treatment of respiratory system infections, airway diseases, and lung tumors.",
        "focus_areas": ["COPD", "Asthma Management", "Lung Nodules", "Pulmonary Infections"],
        "thinking_patterns": ["Lung Function Assessment", "Imaging Sign Interpretation", "Infection vs. Tumor Differentiation", "Airway Reactivity Analysis"],
    },
    "Emergency Medicine": {
        "id": "emergency",
        "name": "Emergency Expert",
        "specialty": "Emergency Medicine",
        "description": "Specializes in initial assessment, emergency resuscitation, and triage of acute illnesses, trauma, and various critical conditions.",
        "focus_areas": ["Vital Signs Maintenance", "Acute Poisoning", "Multiple Trauma", "CPR"],
        "thinking_patterns": ["ABCDE Assessment", "Critical Illness Identification", "Rapid Triage", "Time Sensitivity Judgment"],
    },
    "Urology": {
        "id": "urology",
        "name": "Urology Expert",
        "specialty": "Urology",
        "description": "Specializes in surgical and minimally invasive treatment of the urinary system (kidneys, bladder, urinary tract) and male reproductive system diseases.",
        "focus_areas": ["Urinary Calculi", "Prostate Diseases", "Urinary Tract Tumors", "Urinary Tract Infections"],
        "thinking_patterns": ["Urinalysis Interpretation", "PSA Analysis", "Stone Composition Analysis", "Voiding Function Assessment"],
    },
    "General Practice": {
        "id": "general_practice",
        "name": "General Practice Expert",
        "specialty": "General Practice",
        "description": "Provides comprehensive and continuous basic medical services, specializing in the initial diagnosis of undifferentiated diseases and common disease management.",
        "focus_areas": ["Health Check-up Interpretation", "Common Disease Initial Diagnosis", "Two-way Referral", "Chronic Disease Follow-up"],
        "thinking_patterns": ["Holistic Assessment", "Risk Stratification", "Preventive Medicine", "Multi-system Integration"],
    },
    "Gastrointestinal Surgery": {
        "id": "gastro_surgery",
        "name": "Gastrointestinal Surgery Expert",
        "specialty": "Gastrointestinal Surgery",
        "description": "Specializes in surgical treatment of stomach, small intestine, colorectal, and anal diseases, especially in handling tumors and acute abdomen.",
        "focus_areas": ["Gastrointestinal Tumors", "Appendicitis", "Intestinal Obstruction", "Hernia Repair"],
        "thinking_patterns": ["Acute Abdomen Differentiation", "Intestinal Obstruction Classification", "Tumor Staging", "Surgical Timing Judgment"],
    },
    "Cardiothoracic Surgery": {
        "id": "cardiothoracic_surgery",
        "name": "Cardiothoracic Surgery Expert",
        "specialty": "Cardiothoracic Surgery",
        "description": "Specializes in complex surgical treatment for intrathoracic organs (heart, great vessels, lungs, esophagus).",
        "focus_areas": ["Lung Cancer Surgery", "Heart Valve Disease", "CABG", "Aortic Dissection"],
        "thinking_patterns": ["Chest X-ray/CT Interpretation", "Cardiac Function Assessment", "Surgical Risk Stratification", "Preoperative Optimization"],
    },
    "Oncology": {
        "id": "oncology",
        "name": "Oncology Expert",
        "specialty": "Oncology",
        "description": "Specializes in comprehensive medical treatment of various benign and malignant tumors, including chemotherapy, targeted therapy, and immunotherapy.",
        "focus_areas": ["Chemo/Radiotherapy Plans", "Cancer Screening", "Cancer Pain Management", "MDT"],
        "thinking_patterns": ["Tumor Marker Interpretation", "TNM Staging", "Gene Mutation Analysis", "Prognostic Assessment"],
    },
    "Cardiology": {
        "id": "cardiology",
        "name": "Cardiology Expert",
        "specialty": "Cardiology",
        "description": "Specializes in medical diagnosis, treatment, and interventional therapy of heart and vascular diseases, focusing on cardiovascular risk prevention and control.",
        "focus_areas": ["Hypertension", "Coronary Heart Disease", "Arrhythmia", "Heart Failure"],
        "thinking_patterns": ["ECG Analysis", "Cardiac Enzyme Interpretation", "Risk Stratification", "Anticoagulation Strategy"],
    },
    "Rheumatology": {
        "id": "rheumatology",
        "name": "Rheumatology Expert",
        "specialty": "Rheumatology",
        "description": "Specializes in diagnosis and long-term management of various rheumatic diseases and autoimmune diseases.",
        "focus_areas": ["Rheumatoid Arthritis", "SLE", "Ankylosing Spondylitis", "Gout"],
        "thinking_patterns": ["Autoantibody Profile Analysis", "Multi-system Involvement Assessment", "Immunomodulatory Strategy", "Inflammatory Indicator Monitoring"],
    },
    "Neurology": {
        "id": "neurology",
        "name": "Neurology Expert",
        "specialty": "Neurology",
        "description": "Specializes in medical diagnosis and treatment of central nervous system, peripheral nervous system, and skeletal muscle diseases.",
        "focus_areas": ["Cerebrovascular Disease", "Epilepsy", "Parkinson's Disease", "Peripheral Neuropathy"],
        "thinking_patterns": ["Neurological Examination", "Localization and Qualitative Diagnosis", "Brain Imaging Analysis", "Cognitive Function Assessment"],
    },
}

# List of 16 departments
SPECIALTIES = list(EXPERT_TEMPLATES.keys())

# Department name mapping (handling possible aliases in data)
DEPT_ALIASES = {
    "Obstetrics and Gynecology": ["Obstetrics and Gynecology", "Gynecology", "Obstetrics"],
    "Gastroenterology": ["Gastroenterology", "Gastro", "GI"],
    "Pediatrics": ["Pediatrics", "Peds"],
    "Endocrinology": ["Endocrinology", "Endo"],
    "Hepatobiliary Surgery": ["Hepatobiliary Surgery", "Hepatobiliary"],
    "Orthopedics": ["Orthopedics", "Ortho"],
    "Respiratory Medicine": ["Respiratory Medicine", "Respiratory", "Pulmonology"],
    "Emergency Medicine": ["Emergency Medicine", "Emergency", "ER"],
    "Urology": ["Urology", "Uro"],
    "General Practice": ["General Practice", "GP"],
    "Gastrointestinal Surgery": ["Gastrointestinal Surgery", "GI Surgery"],
    "Cardiothoracic Surgery": ["Cardiothoracic Surgery", "Chest Surgery"],
    "Oncology": ["Oncology", "Cancer"],
    "Cardiology": ["Cardiology", "Heart"],
    "Rheumatology": ["Rheumatology", "Rheuma"],
    "Neurology": ["Neurology", "Neuro"],
}


def load_patients_by_specialty(df: pd.DataFrame, specialty: str, count: int = 5) -> List[PatientInfo]:
    """
    Filters patients of a specified department from a DataFrame
    
    Args:
        df: Patient data DataFrame
        specialty: Department name
        count: Number of cases needed
    
    Returns:
        List of PatientInfo for that department
    """
    aliases = DEPT_ALIASES.get(specialty, [specialty])
    
    # Filter valid cases for the department
    valid_mask = df['is_history_cleaned'] == True
    dept_mask = df['normalized_name'].apply(lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False)
    
    filtered_df = df[valid_mask & dept_mask]
    
    if len(filtered_df) == 0:
        print(f"  ⚠️ Department '{specialty}' not found, trying fuzzy match...")
        # Try looser matching
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
            print(f"    Failed to parse patient: {e}")
            continue
    
    return patients


class SpecialtyPromptOptimizer(GeneticPromptOptimizer):
    """
    Specialty-oriented Prompt Optimizer
    
    Inherits from the general optimizer but:
    1. Uses the department's expert template for evaluation
    2. Outputs structured data containing complete expert settings
    """
    
    def __init__(self, specialty: str, expert_template: Dict, 
                 base_prompt: str, pipeline: DiagnosticPipeline,
                 population_size: int = 16, elitism_count: int = 2):
        super().__init__(base_prompt, pipeline, population_size, elitism_count)
        self.specialty = specialty
        self.expert_template = expert_template
        
    def _get_matched_expert(self, patient: PatientInfo) -> Dict:
        """Override: Always returns the department's expert template"""
        return self.expert_template
    
    def _mutate_prompt(self, prompt_text: str, intensity: str = "medium", 
                      max_retries: int = 2) -> str:
        """Override: Integrates specialty features during mutation"""
        mutation_prompt = f"""
        You are a medical AI prompt optimization expert. Please perform a [Mutation] on the following system prompt for [{self.specialty}] diagnosis.

        [Expert Features]
        - Name: {self.expert_template['name']}
        - Specialty: {self.expert_template['specialty']}
        - Description: {self.expert_template['description']}
        - Focus Areas: {', '.join(self.expert_template['focus_areas'])}
        - Thinking Patterns: {', '.join(self.expert_template.get('thinking_patterns', []))}

        [Original Prompt]
        {prompt_text}

        [Requirements]
        1. Keep all formatting placeholders unchanged ({{resource}}, {{patient}}, {{expert}}, {{reference}}).
        2. Mutation Intensity: {intensity}
        3. Incorporate [{self.specialty}] specialty characteristics to strengthen diagnostic thinking in this field.
        4. Adjustments can include:
           - Diagnostic strategy focus (e.g., more focus on {self.expert_template['focus_areas'][0]}, etc.)
           - Reasoning style (e.g., more rigorous/comprehensive/risk-oriented, etc.)
           - Detailed requirements for output structure
        5. Directly output the modified prompt content, without any explanation or Markdown tags.
        """
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(mutation_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
            except Exception as e:
                print(f"    Warning: Mutation failed (Attempt {attempt+1}/{max_retries}): {e}")
        
        return prompt_text

    def _crossover_prompts(self, prompt_a: str, prompt_b: str, 
                          max_retries: int = 2) -> str:
        """Override: Maintains specialty features during crossover"""
        crossover_prompt = f"""
        You are a medical AI prompt optimization expert. Please [Crossover and Merge] the following two [{self.specialty}] diagnosis prompts.

        [Expert Features]
        - Name: {self.expert_template['name']}
        - Focus Areas: {', '.join(self.expert_template['focus_areas'])}
        - Thinking Patterns: {', '.join(self.expert_template.get('thinking_patterns', []))}

        [Parent Prompt A]
        {prompt_a}

        [Parent Prompt B]
        {prompt_b}

        [Requirements]
        1. Extract best instructional parts from both for combination.
        2. Must keep all formatting placeholders ({{resource}}, {{patient}}, {{expert}}, {{reference}}).
        3. Ensure the merged prompt maintains the [{self.specialty}] specialty characteristics.
        4. Generate a logically consistent and clear new prompt.
        5. Directly output the new prompt content, without any explanation.
        """
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(crossover_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
            except Exception as e:
                print(f"    Warning: Crossover failed (Attempt {attempt+1}/{max_retries}): {e}")
        
        import random
        return random.choice([prompt_a, prompt_b])

    def evaluate_fitness(self, validation_patients: List[PatientInfo], 
                         sample_size: int = 5) -> None:
        """
        Override: Evaluates expert fitness in the MOA multi-disciplinary consultation process
        
        No longer evaluates individual expert diagnosis accuracy in isolation, but instead evaluates:
        The contribution of this expert (using evolved prompt) to the final Step-4 aggregation conclusion after participating in a multi-disciplinary consultation.
        """
        print(f"\nStarting generation {self.generation} fitness evaluation (Sample size: {sample_size}) [MOA Mode]...")
        
        # 1. Stratified sampling (same as parent)
        eval_batch = validation_patients[:sample_size]
        
        # 2. Pre-calculate "consultation context" for each case
        # i.e., opinions of other experts for this case besides the one being optimized
        # This is shared for all individuals in the same generation, only needs calculation once
        print("  - Pre-calculating consultation context (Step 1 & Step 2/3 for other experts)...")
        context_cache = {} # {patient_id: {'other_opinions': [], 'step1': ...}}
        
        for patient in eval_batch:
            try:
                # Run Step 1 routing
                step1_res = self.pipeline.step1_route(patient)
                
                # Get recommended experts list
                recommended_experts = self.pipeline._activate_experts_step1(step1_res["output"])
                
                # Ensure the current specialty being optimized is in the list (if not, force it in)
                # Note: We need to distinguish between "opponents" or "partners"
                other_experts = []
                current_specialty_expert = None
                
                for exp in recommended_experts:
                    if self.expert_template['id'] == exp['id']:
                        current_specialty_expert = exp
                    else:
                        other_experts.append(exp)
                
                # If Step 1 didn't recommend the current specialty, force it to participate
                if not current_specialty_expert:
                    current_specialty_expert = self.expert_template
                
                # Run Step 2 & 3 for other experts (as fixed background)
                other_opinions = []
                for other_exp in other_experts:
                    # Step 2
                    s2 = self.pipeline.step2_semantic_rewrite(patient, other_exp)
                    # Step 3 (using default prompt)
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
                print(f"    Warning: Context pre-calculation failed for {patient.patient_id}: {e}")
                continue
                
        # 3. Evaluate each individual (prompt variant)
        for idx, individual in enumerate(self.population):
            if individual['fitness'] > 0:
                continue
                
            print(f"  - Evaluating individual {idx+1}/{self.population_size} (ID: {individual['id']})...")
            
            total_score = 0
            correct_count = 0
            valid_evaluations = 0
            
            for patient in eval_batch:
                if patient.patient_id not in context_cache:
                    continue
                    
                context = context_cache[patient.patient_id]
                target_expert = context["target_expert_base"]
                
                try:
                    # === Run Step 2 & 3 for current individual ===
                    # Step 2 (semantic rewrite) - simplified by re-running
                    step2_res = self.pipeline.step2_semantic_rewrite(patient, target_expert)
                    
                    # Step 3 (using evolved prompt)
                    step3_res = self.pipeline.step3_diagnosis(
                        patient, target_expert, 
                        step2_res["output"], 
                        custom_prompt_template=individual['prompt'],
                        auto_retrieve=False
                    )
                    
                    # Build current expert's opinion record
                    current_opinion = {
                        "expert_id": target_expert['id'],
                        "expert_name": target_expert['name'],
                        "specialty": target_expert['specialty'],
                        "rewrite": step2_res["output"],
                        "diagnosis": step3_res["output"],
                        "output": step3_res["output"]
                    }
                    
                    # === Combine all expert opinions ===
                    all_opinions = context["other_opinions"] + [current_opinion]
                    
                    # === Run Step 4 (MOA Aggregation) ===
                    step4_res = self.pipeline.step4_aggregate(patient, all_opinions)
                    
                    # === Evaluate final result ===
                    # Key point: evaluating Step 4 aggregation result, not single expert result
                    score, is_correct = self._judge_diagnosis(patient, step4_res['output'])
                    
                    total_score += score
                    if is_correct:
                        correct_count += 1
                    valid_evaluations += 1
                    
                except Exception as e:
                    print(f"    Warning: Evaluation failed for {patient.patient_id}: {e}")
                    continue
            
            if valid_evaluations == 0:
                individual['fitness'] = 0.0
                individual['stats'] = {'accuracy': 0.0, 'avg_score': 0.0, 'valid_count': 0}
                continue
            
            # Calculate average score
            avg_score = total_score / valid_evaluations
            accuracy = correct_count / valid_evaluations
            
            # Fitness formula
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
        """Runs evolution and returns results with complete expert settings"""
        # Call parent evolution
        raw_results = super().run_evolution(validation_patients, generations, top_k_return)
        
        # Enrich results with complete expert settings
        enriched_results = []
        for i, result in enumerate(raw_results):
            expert_data = {
                # Expert identity info
                "id": f"{self.expert_template['id']}_{i+1}",
                "name": f"{self.expert_template['name']} (Variant {i+1})",
                "specialty": self.specialty,
                "description": self.expert_template['description'],
                "focus_areas": self.expert_template['focus_areas'],
                "thinking_patterns": self.expert_template.get('thinking_patterns', []),
                # Evolved prompt
                "evolved_prompt": result['prompt'],
                # Evaluation metrics
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
    Runs GA evolution for a single department
    
    Args:
        specialty: Department name
        patients: Validation cases for the department
        pipeline: Diagnosis pipeline
        generations: Evolution generations
        population_size: Population size
        top_k_return: Number of best experts to return
    
    Returns:
        Top K experts for the department (including complete settings)
    """
    print(f"\n{'='*70}")
    print(f"🧬 Starting Evolution: [{specialty}] Expert Pool")
    print(f"  - Validation Cases: {len(patients)}")
    print(f"  - Population Size: {population_size}")
    print(f"  - Generations: {generations}")
    print(f"{'='*70}")
    
    if len(patients) == 0:
        print(f"  ❌ No valid cases, skipping department: {specialty}")
        return []
    
    # Get expert template for the department
    expert_template = EXPERT_TEMPLATES.get(specialty)
    if not expert_template:
        print(f"  ❌ Department template not found: {specialty}")
        return []
    
    print(f"  📋 Expert Template: {expert_template['name']}")
    print(f"     Focus Areas: {', '.join(expert_template['focus_areas'])}")
    print(f"     Thinking Patterns: {', '.join(expert_template.get('thinking_patterns', []))}")
    
    # Use specialty-oriented optimizer
    optimizer = SpecialtyPromptOptimizer(
        specialty=specialty,
        expert_template=expert_template,
        base_prompt=system_step3_prompt,
        pipeline=pipeline,
        population_size=population_size,
        elitism_count=max(2, population_size // 8)
    )
    
    # Run evolution
    best_experts = optimizer.run_evolution(
        validation_patients=patients,
        generations=generations,
        top_k_return=top_k_return
    )
    
    print(f"\n✅ [{specialty}] Evolution complete, obtained {len(best_experts)} optimized experts")
    for i, expert in enumerate(best_experts):
        print(f"   [{i+1}] {expert['name']} - Fitness: {expert.get('fitness', 0):.4f}")
    
    return best_experts


def main():
    """Main function: Specialty training + Merging Expert Pool"""
    
    print("="*70)
    print("🏥 Specialty Track Genetic Algorithm Evolution System")
    print("="*70)
    print(f"Target Departments: {len(SPECIALTIES)}")
    print(f"Cases per Dept: 5")
    print(f"Generations per Dept: 10")
    print(f"Output per Dept: 4 Best Experts")
    print(f"Final Expert Pool: {len(SPECIALTIES) * 4} members")
    print("="*70)
    
    # Load data
    print("\n📂 Loading patient data...")
    data_file = "guilin_inpatient_extracted_10000(2).xlsx"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    df = pd.read_excel(data_file)
    print(f"Loaded {len(df)} patient records")
    
    # Stat patient counts for each department
    print("\n📊 Patient Distribution by Department:")
    for specialty in SPECIALTIES:
        aliases = DEPT_ALIASES.get(specialty, [specialty])
        count = df['normalized_name'].apply(
            lambda x: any(alias in str(x) for alias in aliases) if pd.notna(x) else False
        ).sum()
        print(f"  - {specialty}: {count} cases")
    
    # Initialize Diagnosis Pipeline (shared to save resources)
    print("\n🔧 Initializing Diagnosis Pipeline...")
    pipeline = DiagnosticPipeline(
        activation_mode="eep_semantic",
        enable_rag=False,       # Disable RAG for faster training
        enable_experience=False,
        enable_case=False,
    )
    
    # Specialty training
    all_experts = []
    failed_specialties = []
    
    for idx, specialty in enumerate(SPECIALTIES):
        print(f"\n\n{'#'*70}")
        print(f"# Progress: [{idx+1}/{len(SPECIALTIES)}] {specialty}")
        print(f"{'#'*70}")
        
        # Get validation cases for the department
        patients = load_patients_by_specialty(df, specialty, count=5)
        print(f"  Filtered {len(patients)} cases")
        
        if len(patients) < 3:
            print(f"  ⚠️ Less than 3 cases, skipping department: {specialty}")
            failed_specialties.append(specialty)
            continue
        
        # Run evolution
        try:
            experts = run_specialty_evolution(
                specialty=specialty,
                patients=patients,
                pipeline=pipeline,
                generations=10,
                population_size=16,  # Small population to save cost
                top_k_return=4
            )
            all_experts.extend(experts)
            
            # Save intermediate results (prevent loss if crash)
            intermediate_file = f"moa_specialty_pool_{specialty}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(experts, f, ensure_ascii=False, indent=2)
            print(f"  💾 Intermediate result saved: {intermediate_file}")
            
        except Exception as e:
            print(f"  ❌ Training failed: {e}")
            failed_specialties.append(specialty)
            continue
        
        # Brief rest to avoid API rate limit
        time.sleep(2)
    
    # Merge final expert pool
    print("\n\n" + "="*70)
    print("🏆 Merging Final Expert Pool")
    print("="*70)
    
    # Sort by fitness
    all_experts.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    
    # Save final results
    output_file = "moa_optimized_expert_pool_64.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Final expert pool saved: {output_file}")
    print(f"   Total {len(all_experts)} experts")
    
    # Summary report
    print("\n📋 Training Summary:")
    print(f"  - Successful Departments: {len(SPECIALTIES) - len(failed_specialties)}")
    if failed_specialties:
        print(f"  - Failed Departments: {failed_specialties}")
    
    print("\n🎉 Expert Pool Distribution:")
    specialty_counts = {}
    for expert in all_experts:
        s = expert.get('specialty', 'Unknown')
        specialty_counts[s] = specialty_counts.get(s, 0) + 1
    
    for s, c in specialty_counts.items():
        print(f"  - {s}: {c} experts")
    
    # Display expert details
    print("\n📋 Expert Details:")
    for i, expert in enumerate(all_experts):
        print(f"  [{i+1:2d}] {expert.get('name', 'Unknown')}")
        print(f"       Specialty: {expert.get('specialty', 'Unknown')}")
        print(f"       Focus: {', '.join(expert.get('focus_areas', []))}")
        print(f"       Fitness: {expert.get('fitness', 0):.4f}")
    
    # Clean up intermediate files (optional)
    # for specialty in SPECIALTIES:
    #     intermediate_file = f"specialty_pool_{specialty}.json"
    #     if os.path.exists(intermediate_file):
    #         os.remove(intermediate_file)
    
    return all_experts


if __name__ == "__main__":
    main()
