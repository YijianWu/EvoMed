# -*- coding: utf-8 -*-
"""Merges JSON from batch_results into Excel"""

import json
import os
import pandas as pd

batch_dir = 'batch_results'
json_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.json')])
print(f'Found {len(json_files)} JSON files')

rows = []
for f in json_files:
    with open(os.path.join(batch_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    steps = data.get('steps', {})
    routing = steps.get('routing', {})
    step1 = steps.get('step1', {})
    step4 = steps.get('step4', {})
    evolution = steps.get('expert_evolution', {})
    
    # Extract expert diagnosis opinions
    expert_opinions = steps.get('expert_opinions', [])
    expert_diag_list = []
    for op in expert_opinions:
        name = op.get('expert_name', '')
        diag = op.get('diagnosis', '')
        diag_short = diag[:800] if diag else ''
        expert_diag_list.append(f"【{name}】\n{diag_short}")
    
    row = {
        'Case No.': f.replace('.json', ''),
        'Patient ID': data.get('patient_id', ''),
        'Activation Mode': data.get('activation_mode', ''),
        'Episode ID': routing.get('episode_id', ''),
        'Episode Keywords': ', '.join(routing.get('keywords', [])),
        'System Domain': ', '.join(routing.get('system_domain', [])),
        'Step 1 Recommended Specialties': ', '.join(routing.get('step1_recommended_specialties', [])),
        'Activated Experts': ', '.join(routing.get('selected_experts', [])),
        'Step 1 Routing Suggestion': step1.get('output', ''),
        'Individual Expert Opinions': '\n\n'.join(expert_diag_list),
        'Step 4 Comprehensive Diagnosis': step4.get('output', ''),
        'New Expert Generated': 'Yes' if evolution.get('new_expert_generated') else 'No',
        'New Expert Name': evolution.get('new_expert_name', ''),
        'Divergence Point': evolution.get('divergence_point', ''),
    }
    
    rows.append(row)

df = pd.DataFrame(rows)
output_file = 'diagnosis_results.xlsx'

# Use openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
from openpyxl.styles import Alignment

wb = Workbook()
ws = wb.active
ws.title = 'Diagnosis Results'

# Write data
for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = Alignment(wrap_text=True, vertical='top')

# Set column widths
col_widths = [12, 15, 12, 15, 30, 25, 40, 50, 80, 100, 100, 10, 20, 40]
for i, width in enumerate(col_widths, 1):
    ws.column_dimensions[chr(64 + i)].width = min(width, 50)

wb.save(output_file)

print(f'Saved to: {output_file}')
print(f'Total {len(df)} records')
print('\nColumn Info:')
for col in df.columns:
    print(f'  - {col}')
