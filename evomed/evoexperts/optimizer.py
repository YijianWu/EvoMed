"""
Genetic Algorithm Optimizer: Evolves system prompts for diagnostic agents
"""

import json
import random
from typing import Dict, List, Optional, Any, Tuple
from evomed.diagnosis import PatientInfo, DiagnosticPipeline

class GeneticPromptOptimizer:
    """
    Genetic Algorithm Optimizer: Evolves system prompts for diagnostic agents
    """
    
    def __init__(self, base_prompt: str, pipeline: DiagnosticPipeline, 
                 population_size: int = 32, elitism_count: int = 6):
        self.base_prompt = base_prompt
        self.pipeline = pipeline
        self.client = pipeline.client
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.hall_of_fame = []
        self.step2_cache = {}
        self.expert_match_cache = {}
        self.fitness_history = []
        self.stagnation_count = 0
        self.max_stagnation = 3
        
    def _get_matched_expert(self, patient: PatientInfo) -> Dict:
        """Intelligent expert matching based on patient department and symptoms"""
        if patient.patient_id in self.expert_match_cache:
            return self.expert_match_cache[patient.patient_id]
            
        department = patient.department.lower()
        for expert in self.pipeline.experts:
            if any(dept in department for dept in [expert['specialty'].lower(), expert['name'].lower()]):
                self.expert_match_cache[patient.patient_id] = expert
                return expert
        
        for expert in self.pipeline.experts:
            if expert['id'] == 'general_practice':
                self.expert_match_cache[patient.patient_id] = expert
                return expert
                
        self.expert_match_cache[patient.patient_id] = self.pipeline.experts[0]
        return self.pipeline.experts[0]
        
    def initialize_population(self):
        """Initializes population with diverse variants of the base prompt"""
        print(f"Initializing population (Size: {self.population_size})...")
        
        self.population.append({
            'id': 'gen0_original',
            'prompt': self.base_prompt,
            'fitness': 0.0,
            'stats': {}
        })
        
        for i in range(1, self.population_size):
            try:
                print(f"  - Generating initial individual {i}/{self.population_size}...")
                mutated_prompt = self._mutate_prompt(self.base_prompt, intensity="high")
                self.population.append({
                    'id': f'gen0_ind{i}',
                    'prompt': mutated_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
            except Exception as e:
                print(f"    Warning: Failed to generate individual {i}: {e}, using original prompt")
                self.population.append({
                    'id': f'gen0_ind{i}_fallback',
                    'prompt': self.base_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
                
    def evaluate_fitness(self, validation_patients: List[PatientInfo], 
                         sample_size: int = 5) -> None:
        """Evaluates population fitness using stratified sampling and caching"""
        print(f"\nStarting fitness evaluation for generation {self.generation} (Sample size: {sample_size})...")
        
        dept_groups = {}
        for p in validation_patients:
            dept = p.department
            if dept not in dept_groups:
                dept_groups[dept] = []
            dept_groups[dept].append(p)
            
        eval_batch = []
        depts = list(dept_groups.keys())
        random.shuffle(depts)
        
        while len(eval_batch) < sample_size and len(depts) > 0:
            for dept in depts[:]:
                if len(dept_groups[dept]) > 0:
                    p = random.choice(dept_groups[dept])
                    if p not in eval_batch:
                        eval_batch.append(p)
                        dept_groups[dept].remove(p) 
                else:
                    depts.remove(dept)
                
                if len(eval_batch) >= sample_size:
                    break
        
        print(f"  - Validation set composition: {', '.join([p.department for p in eval_batch])}")
        
        print("  - Pre-calculating Step-2 results (Cache optimization)...")
        for patient in eval_batch:
            expert = self._get_matched_expert(patient)
            cache_key = (patient.patient_id, expert['id'])
            
            if cache_key not in self.step2_cache:
                try:
                    step2_res = self.pipeline.step2_semantic_rewrite(patient, expert)
                    self.step2_cache[cache_key] = step2_res['output']
                except Exception as e:
                    print(f"    Warning: Step-2 failed for {patient.patient_id}: {e}")
                    self.step2_cache[cache_key] = f"Rewriting failed, original info: {patient.to_prompt_string()[:500]}"
        
        for idx, individual in enumerate(self.population):
            if individual['fitness'] > 0:
                continue
                
            print(f"  - Evaluating individual {idx+1}/{self.population_size} (ID: {individual['id']})...")
            
            total_score = 0
            correct_count = 0
            valid_evaluations = 0
            
            for patient in eval_batch:
                try:
                    expert = self._get_matched_expert(patient)
                    cache_key = (patient.patient_id, expert['id'])
                    step2_output = self.step2_cache.get(cache_key, "")
                    
                    step3_res = self.pipeline.step3_diagnosis(
                        patient, expert, 
                        step2_output, 
                        custom_prompt_template=individual['prompt'],
                        auto_retrieve=False
                    )
                    
                    score, is_correct = self._judge_diagnosis(patient, step3_res['output'])
                    total_score += score
                    if is_correct:
                        correct_count += 1
                    valid_evaluations += 1
                        
                except Exception as e:
                    print(f"    Warning: Evaluation failed for patient {patient.patient_id}: {e}")
                    continue
            
            if valid_evaluations == 0:
                individual['fitness'] = 0.0
                individual['stats'] = {'accuracy': 0.0, 'avg_score': 0.0, 'valid_count': 0}
                continue
                
            avg_score = total_score / valid_evaluations
            accuracy = correct_count / valid_evaluations
            fitness = accuracy * 0.7 + (avg_score / 100) * 0.3
            if valid_evaluations < len(eval_batch) * 0.5:
                fitness *= 0.8
            
            individual['fitness'] = fitness
            individual['stats'] = {
                'accuracy': accuracy, 
                'avg_score': avg_score, 
                'valid_count': valid_evaluations
            }
            print(f"    -> Fitness: {fitness:.4f} (Acc: {accuracy:.0%}, Valid: {valid_evaluations}/{len(eval_batch)})")
            
    def _judge_diagnosis(self, patient: PatientInfo, diagnosis_output: str, 
                        max_retries: int = 2) -> Tuple[float, bool]:
        """Uses LLM as a judge to evaluate diagnostic accuracy"""
        judge_prompt = f\"\"\"
You are a senior medical expert judge. Please evaluate the diagnostic accuracy of the AI doctor.

[Golden Standard / True Diagnosis]
{patient.main_diagnosis} (ICD: {patient.main_diagnosis_icd})

[AI Doctor Diagnostic Output]
{diagnosis_output}

Please rate (0-100) and determine if it is correct (True/False).
Criteria:
- 100: Completely consistent, core diagnosis correct.
- 80-99: Core diagnosis correct, minor details missing.
- 60-79: Correct diagnostic direction, but specific disease is slightly off.
- 0-59: Misdiagnosis or missed diagnosis.

Output JSON: {{"score": 85, "is_correct": true, "reason": "..."}}
Output JSON only.\"\"\"
        
        for attempt in range(max_retries):
            try:
                res = self.pipeline._call_llm(judge_prompt)
                if "```" in res:
                    res = res.split("```json")[-1].split("```")[0] if "```json" in res else res.split("```")[-1].split("```")[0]
                
                data = json.loads(res.strip())
                score = float(data.get('score', 0))
                is_correct = bool(data.get('is_correct', False))
                
                if 0 <= score <= 100:
                    return score, is_correct
                else:
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
        
        return 30.0, False

    def _mutate_prompt(self, prompt_text: str, intensity: str = "medium", 
                      max_retries: int = 2) -> str:
        """Mutation operator: Uses LLM to modify the prompt"""
        mutation_prompt = f\"\"\"
You are a prompt optimization expert. Please perform a [Mutation] on the following system prompt used for medical diagnosis.
Mutation Intensity: {intensity}

[Original Prompt]
{prompt_text}

[Requirements]
1. Keep all formatting placeholders unchanged (e.g., {{resource}}, {{patient}}, {{expert}}).
2. Randomly alter diagnostic strategy (e.g., more aggressive, more conservative, focus more on history, etc.).
3. Adjust phrasing slightly to elicit different potentials of the LLM.
4. Directly output the modified prompt content, without any explanation or Markdown tags.
\"\"\"
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(mutation_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
                else:
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
        
        return prompt_text

    def _crossover_prompts(self, prompt_a: str, prompt_b: str, 
                          max_retries: int = 2) -> str:
        """Crossover operator: Merges two prompts"""
        crossover_prompt = f\"\"\"
You are a prompt optimization expert. Please merge the following two medical diagnosis prompts into a new, superior prompt.

[Parent Prompt A]
{prompt_a}

[Parent Prompt B]
{prompt_b}

[Requirements]
1. Extract the best instructional parts from both for combination.
2. Must keep all formatting placeholders ({{resource}}, {{patient}}, etc.).
3. Generate a logically consistent and clear new prompt.
4. Directly output the new prompt content, without any explanation.
\"\"\"
        
        for attempt in range(max_retries):
            try:
                result = self.pipeline._call_llm(crossover_prompt).strip()
                required_placeholders = ['{resource}', '{patient}', '{expert}']
                if all(ph in result for ph in required_placeholders):
                    return result
                else:
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
        
        return random.choice([prompt_a, prompt_b])

    def selection(self):
        """Selection operator: Elitism + Roulette Wheel"""
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        current_best = sorted_pop[0]['fitness']
        
        if len(self.fitness_history) > 0:
            if current_best <= max(self.fitness_history):
                self.stagnation_count += 1
                print(f"  - No improvement ({self.stagnation_count}/{self.max_stagnation})")
            else:
                self.stagnation_count = 0
        
        self.fitness_history.append(current_best)
        self.best_individual = sorted_pop[0]
        print(f"  - Current best individual: {self.best_individual['id']}, Fitness: {current_best:.4f}")
        
        new_pop = []
        for i in range(min(self.elitism_count, len(sorted_pop))):
            new_pop.append(sorted_pop[i])
            print(f"    Keeping elite: {sorted_pop[i]['id']} (Fitness: {sorted_pop[i]['fitness']:.4f})")
        
        parents_pool = sorted_pop[:max(len(sorted_pop)//2, 2)]
        while len(new_pop) < self.population_size:
            if random.random() < 0.8 and len(parents_pool) >= 2:
                parent_a = random.choice(parents_pool)
                parent_b = random.choice(parents_pool)
                if parent_a == parent_b and len(parents_pool) > 1:
                    continue
                
                child_prompt = self._crossover_prompts(parent_a['prompt'], parent_b['prompt'])
                if random.random() < 0.1:
                    child_prompt = self._mutate_prompt(child_prompt, intensity="low")
                
                new_pop.append({
                    'id': f'gen{self.generation+1}_child_{len(new_pop)}',
                    'prompt': child_prompt,
                    'fitness': 0.0,
                    'stats': {}
                })
            else:
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
        """Checks if early stopping should be triggered"""
        return self.stagnation_count >= self.max_stagnation

    def run_evolution(self, validation_patients: List[PatientInfo], generations: int = 10, top_k_return: int = 32):
        """Runs the main evolution loop"""
        self.initialize_population()
        
        for g in range(generations):
            self.generation = g
            print(f"\n" + "="*60)
            print(f"[Evolution Generation: {g+1}/{generations}]")
            print(f"="*60)
            
            self.evaluate_fitness(validation_patients)
            
            for ind in self.population:
                if ind.get('fitness', 0) > 0:
                    exists = False
                    for existing in self.hall_of_fame:
                        if existing['prompt'] == ind['prompt']:
                            exists = True
                            if ind['fitness'] > existing['fitness']:
                                existing['fitness'] = ind['fitness']
                                existing['stats'] = ind['stats']
                                existing['id'] = ind['id']
                            break
                    if not exists:
                        self.hall_of_fame.append(ind.copy())
            
            self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
            print(f"  📊 Hall of Fame currently has: {len(self.hall_of_fame)} valid experts")
            
            best = max(self.population, key=lambda x: x['fitness'])
            print(f"\n>>> Best Prompt of Gen {g+1} (Fitness: {best['fitness']:.4f}):")
            print(best['prompt'][:200] + "...")
            
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
            
            if g < generations - 1:
                self.selection()
                if self.should_early_stop():
                    print(f"\n🔄 Early stop triggered: {self.stagnation_count} generations without improvement")
                    break
                
        print("\n🏆 Evolution complete! Exporting optimal expert pool...")
        final_pool = self.hall_of_fame[:top_k_return]
        
        if len(final_pool) < top_k_return:
            seen_prompts = {p['prompt'] for p in final_pool}
            for ind in self.population:
                if len(final_pool) >= top_k_return:
                    break
                if ind['prompt'] not in seen_prompts:
                    final_pool.append(ind)
                    seen_prompts.add(ind['prompt'])
        
        return final_pool

