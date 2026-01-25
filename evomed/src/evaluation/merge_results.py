# -*- coding: utf-8 -*-
"""合并batch_results中的JSON到Excel"""

import json
import os
import pandas as pd

batch_dir = 'batch_results'
json_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.json')])
print(f'找到 {len(json_files)} 个JSON文件')

rows = []
for f in json_files:
    with open(os.path.join(batch_dir, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    steps = data.get('steps', {})
    routing = steps.get('routing', {})
    step1 = steps.get('step1', {})
    step4 = steps.get('step4', {})
    evolution = steps.get('expert_evolution', {})
    
    # 提取ExpertDiagnosis意见
    expert_opinions = steps.get('expert_opinions', [])
    expert_diag_list = []
    for op in expert_opinions:
        name = op.get('expert_name', '')
        diag = op.get('diagnosis', '')
        diag_short = diag[:800] if diag else ''
        expert_diag_list.append(f"【{name}】\n{diag_short}")
    
    row = {
        'Case号': f.replace('.json', ''),
        'PatientID': data.get('patient_id', ''),
        '激活模式': data.get('activation_mode', ''),
        'Episode_ID': routing.get('episode_id', ''),
        'Episode关键词': ', '.join(routing.get('keywords', [])),
        '系统域': ', '.join(routing.get('system_domain', [])),
        'Step1推荐专科': ', '.join(routing.get('step1_recommended_specialties', [])),
        '激活Expert': ', '.join(routing.get('selected_experts', [])),
        'Step1_路由建议': step1.get('output', ''),
        '各ExpertDiagnosis意见': '\n\n'.join(expert_diag_list),
        'Step4_综合Diagnosis': step4.get('output', ''),
        '新Expert生成': '是' if evolution.get('new_expert_generated') else '否',
        '新Expert名称': evolution.get('new_expert_name', ''),
        '分歧点': evolution.get('divergence_point', ''),
    }
    
    rows.append(row)

df = pd.DataFrame(rows)
output_file = 'diagnosis_results.xlsx'

# 使用openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
from openpyxl.styles import Alignment

wb = Workbook()
ws = wb.active
ws.title = 'Diagnosis结果'

# 写入数据
for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = Alignment(wrap_text=True, vertical='top')

# 设置列宽
col_widths = [12, 15, 12, 15, 30, 25, 40, 50, 80, 100, 100, 10, 20, 40]
for i, width in enumerate(col_widths, 1):
    ws.column_dimensions[chr(64 + i)].width = min(width, 50)

wb.save(output_file)

print(f'已保存到: {output_file}')
print(f'共 {len(df)} 条记录')
print('\n列信息:')
for col in df.columns:
    print(f'  - {col}')

