# 数据说明

## 数据要求

本项目需要以下数据文件（未包含在仓库中，需自行准备）：

### 1. 患者数据
- **文件名**: `guilin_inpatient_extracted_10000.xlsx`
- **格式**: Excel
- **必需字段**:
  - `patient_id`: 患者ID
  - `age`: 年龄
  - `gender`: 性别
  - `chief_complaint`: 主诉
  - `present_illness`: 现病史
  - `past_history`: 既往史
  - `physical_exam`: 体格检查
  - `lab_results`: 实验室检查
  - `diagnosis`: 诊断（用于验证）
  - `normalized_name`: 科室名称
  - `is_history_cleaned`: 数据清洗标记

### 2. 数据隐私

由于医疗数据的隐私性和敏感性，本仓库不包含任何真实患者数据。

如需使用本系统，请：
1. 准备符合上述格式的数据集
2. 确保已获得必要的伦理审批
3. 遵守相关数据隐私法规（如HIPAA、GDPR等）

### 3. 示例数据格式

```python
{
    "patient_id": "P001",
    "age": 45,
    "gender": "女",
    "chief_complaint": "腹痛3天",
    "present_illness": "患者3天前无明显诱因出现右上腹痛，呈持续性胀痛...",
    "past_history": "既往体健，否认高血压、糖尿病史",
    "physical_exam": "体温37.2℃，右上腹压痛阳性，墨菲氏征阳性",
    "lab_results": "WBC 12.5×10^9/L，中性粒细胞82%",
    "diagnosis": "急性胆囊炎",
    "normalized_name": "肝胆外科",
    "is_history_cleaned": true
}
```

### 4. 数据预处理

如果您的数据格式不同，可能需要进行预处理：

```python
import pandas as pd

def preprocess_data(input_file, output_file):
    df = pd.read_excel(input_file)
    
    # 数据清洗和格式转换
    # ...
    
    df.to_excel(output_file, index=False)
```

## 注意事项

- 所有医疗数据必须脱敏处理
- 确保数据使用符合伦理规范
- 建议使用合成数据或公开数据集进行测试


